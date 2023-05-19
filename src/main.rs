#![feature(core_intrinsics)]
use std::{
    cmp::{self, max, min},
    f32::MIN,
    fs::OpenOptions,
    intrinsics::{likely, unlikely},
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

use indicatif::{ProgressBar, ProgressStyle};

// 一个全局变量MAX_ORDER，用来表示buddy算法的最大阶数 [MIN_ORDER, MAX_ORDER)左闭右开区间
const MAX_ORDER: usize = 31;
// 4KB
const MIN_ORDER: usize = 12;

// 一个全局变量MAX_PAGES，用来表示buddy算法最多能够管理的页数
// const MAX_PAGES: usize = (2048);
const MAX_PAGES: usize = (1 << 48) / (1 << 12);

static CURRENT_PAGE: AtomicUsize = AtomicUsize::new(0);

// == 计算buddy所需要预分配的页数的代码 BEGIN ==

#[derive(Debug, Clone)]
enum CalculateError {
    PagesError(String),
    EntriesError(String),
    NoEnoughPages(String),
}

struct BuddyPageCalculator {
    layers: [BuddyCalculatorLayer; MAX_ORDER - MIN_ORDER],
    /// 总的页数
    total_pages: usize,
    /// 每个页能够存放的buddy entry的数量
    entries_per_page: usize,
    max_order: usize,
}

macro_rules! calculator_layer {
    ($self: ident, $order: expr) => {
        $self.layers[$order - MIN_ORDER]
    };
}

impl BuddyPageCalculator {
    const PAGE_4K: usize = (1 << 12);
    const PAGE_1G: usize = (1 << 30);
    const MAX_ORDER_SIZE: usize = (1 << (MAX_ORDER - 1));

    const fn new(entries_per_page: usize) -> Self {
        BuddyPageCalculator {
            layers: [BuddyCalculatorLayer::new(); MAX_ORDER - MIN_ORDER],
            total_pages: 0,
            entries_per_page,
            max_order: 0,
        }
    }

    fn calculate(&mut self, pages: usize) -> Result<Vec<BuddyCalculatorResult>, CalculateError> {
        self.total_pages = pages;
        self.init_layers();

        self.sim()?;

        let mut res = Vec::new();
        for order in MIN_ORDER..MAX_ORDER {
            let layer = &calculator_layer!(self, order);
            res.push(BuddyCalculatorResult::new(
                order,
                layer.allocated_pages,
                layer.entries,
            ));
        }
        // print_results(&res);

        return Ok(res);
    }

    fn sim(&mut self) -> Result<(), CalculateError> {
        loop {
            let mut flag = false;
            'outer: for order in (MIN_ORDER..MAX_ORDER).rev() {
                let mut to_alloc =
                    self.pages_need_to_alloc(order, calculator_layer!(self, order).entries);
                // 模拟申请
                while to_alloc > 0 {
                    let page4k = calculator_layer!(self, MIN_ORDER).entries;
                    let page4k = min(page4k, to_alloc);
                    calculator_layer!(self, order).allocated_pages += page4k;
                    calculator_layer!(self, MIN_ORDER).entries -= page4k;
                    to_alloc -= page4k;

                    if to_alloc == 0 {
                        break;
                    }

                    // 从最小的order开始找，然后分裂
                    let split_order = ((MIN_ORDER + 1)..=order).find(|&i| {
                        let layer = &calculator_layer!(self, i);
                        // println!("find: order: {}, entries: {}", i, layer.entries);
                        layer.entries > 0
                    });

                    if let Some(split_order) = split_order {
                        for i in (MIN_ORDER + 1..=split_order).rev() {
                            let layer = &mut calculator_layer!(self, i);
                            layer.entries -= 1;
                            calculator_layer!(self, i - 1).entries += 2;
                        }
                    } else {
                        // 从大的开始分裂
                        let split_order = ((order + 1)..MAX_ORDER).find(|&i| {
                            let layer = &calculator_layer!(self, i);
                            // println!("find: order: {}, entries: {}", i, layer.entries);
                            layer.entries > 0
                        });
                        if let Some(split_order) = split_order {
                            for i in (order + 1..=split_order).rev() {
                                let layer = &mut calculator_layer!(self, i);
                                layer.entries -= 1;
                                calculator_layer!(self, i - 1).entries += 2;
                            }
                            flag = true;
                            break 'outer;
                        } else {
                            if order == MIN_ORDER
                                && to_alloc == 1
                                && calculator_layer!(self, MIN_ORDER).entries > 0
                            {
                                calculator_layer!(self, MIN_ORDER).entries -= 1;
                                calculator_layer!(self, MIN_ORDER).allocated_pages += 1;
                                break;
                            } else {
                                return Err(CalculateError::NoEnoughPages(format!(
                                    "order: {}, pages_needed: {}",
                                    order, to_alloc
                                )));
                            }
                        }
                    }
                }
            }

            if !flag {
                break;
            }
        }
        return Ok(());
    }

    fn init_layers(&mut self) {
        let max_order = min(log2(self.total_pages * Self::PAGE_4K), MAX_ORDER - 1);

        self.max_order = max_order;
        let mut remain_bytes = self.total_pages * Self::PAGE_4K;
        for order in (MIN_ORDER..=max_order).rev() {
            let entries = remain_bytes / (1 << order);
            remain_bytes -= entries * (1 << order);
            calculator_layer!(self, order).entries = entries;
            // println!(
            //     "order: {}, entries: {}, pages: {}",
            //     order,
            //     entries,
            //     calculator_layer!(self, order).allocated_pages
            // );
        }
    }

    fn entries_to_page(&self, entries: usize) -> usize {
        (entries + self.entries_per_page - 1) / self.entries_per_page
    }

    fn pages_needed(&self, entries: usize) -> usize {
        max(1, self.entries_to_page(entries))
    }
    fn pages_need_to_alloc(&self, order: usize, current_entries: usize) -> usize {
        let allocated = calculator_layer!(self, order).allocated_pages;
        let tot_need = self.pages_needed(current_entries);
        if tot_need > allocated {
            tot_need - allocated
        } else {
            0
        }
    }
    fn check_result(&self, results: &Vec<BuddyCalculatorResult>) -> Result<(), CalculateError> {
        // 检查pages是否正确
        let mut total_pages = 0;
        for r in results.iter() {
            total_pages += r.pages;
            total_pages += r.entries * (1 << r.order) / Self::PAGE_4K;
        }
        if unlikely(total_pages != self.total_pages) {
            // println!("total_pages: {}, self.total_pages: {}", total_pages, self.total_pages);
            return Err(CalculateError::PagesError(format!(
                "total_pages: {}, self.total_pages: {}",
                total_pages, self.total_pages
            )));
        }
        // 在确认pages正确的情况下，检查每个链表的entries是否正确
        // 检查entries是否正确
        for r in results.iter() {
            let pages_needed = self.pages_needed(r.entries);
            if pages_needed != r.pages {
                if likely(
                    r.order == (MAX_ORDER - 1)
                        && (pages_needed as isize - r.pages as isize).abs() == 1,
                ) {
                    continue;
                }
                return Err(CalculateError::EntriesError(format!(
                    "order: {}, pages_needed: {}, pages: {}",
                    r.order,
                    self.pages_needed(r.entries),
                    r.pages
                )));
            }
        }
        return Ok(());
    }
}

#[derive(Debug, Clone, Copy)]
struct BuddyCalculatorLayer {
    /// 当前层的buddy entry的数量
    entries: usize,
    allocated_pages: usize,
}

impl BuddyCalculatorLayer {
    const fn new() -> Self {
        BuddyCalculatorLayer {
            entries: 0,
            allocated_pages: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BuddyCalculatorResult {
    order: usize,
    pages: usize,
    entries: usize,
}

impl BuddyCalculatorResult {
    fn new(order: usize, pages: usize, entries: usize) -> Self {
        BuddyCalculatorResult {
            order,
            pages,
            entries,
        }
    }
}

fn print_results(results: &Vec<BuddyCalculatorResult>) {
    let mut total_pages = 0;
    for r in results {
        println!(
            "order: {}, pages: {}, entries: {}",
            r.order, r.pages, r.entries
        );
        total_pages += r.pages;
    }
    println!("total_pages: {}", total_pages);
}
// == 计算buddy所需要预分配的页数的代码 END ==

// 多线程枚举所有的内存大小。

struct Task {
    start_page: usize,
    num_pages: usize,
}

impl Task {
    fn new(start_page: usize, num_pages: usize) -> Self {
        Task {
            start_page,
            num_pages,
        }
    }
}
/// 生成任务
fn task_generator() -> Option<Task> {
    const PAGE_NUM: usize = 512;

    let page = CURRENT_PAGE.fetch_add(PAGE_NUM, Ordering::SeqCst);
    if page >= MAX_PAGES {
        return None;
    }

    return Some(Task::new(page, PAGE_NUM));
}

#[derive(Debug)]
struct ExceptionValue {
    page: usize,
    reason: String,
}

impl PartialEq for ExceptionValue {
    fn eq(&self, other: &Self) -> bool {
        self.page == other.page
    }
}
impl PartialOrd for ExceptionValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.page.cmp(&other.page))
    }
}
impl Eq for ExceptionValue {}
impl Ord for ExceptionValue {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.page.cmp(&other.page)
    }
}

fn worker(entries_per_page: usize) -> Vec<ExceptionValue> {
    let mut exception_value = Vec::new();
    loop {
        let task: Option<Task> = task_generator();
        if task.is_none() {
            break;
        }

        let task = task.unwrap();

        for i in task.start_page..task.start_page + task.num_pages {
            let page = task.start_page + i;
            let mut calculator = BuddyPageCalculator::new(entries_per_page);
            let r = calculator.calculate(page);
            if r.is_err() {
                exception_value.push(ExceptionValue {
                    page,
                    reason: format!("{:?}", r.unwrap_err()),
                });
                continue;
            }
            if let Err(e) = calculator.check_result(&r.unwrap()) {
                exception_value.push(ExceptionValue {
                    page,
                    reason: format!("check_result error:{:?}", e),
                });
            }
        }
    }

    return exception_value;
}

fn simu_show_progress() {
    println!("Simulating...");
    let pb = ProgressBar::new(MAX_PAGES as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ETA: {eta})",
        )
        .unwrap(),
    );
    loop {
        let current = CURRENT_PAGE.load(Ordering::Relaxed);
        if current >= MAX_PAGES {
            break;
        }
        pb.set_position(current as u64);
        // 睡眠1s
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

    pb.finish();
}

fn simulate(entries_per_page: usize) {
    // 获取cpu核心数
    let cpus = num_cpus::get();

    let mut handles = Vec::new();
    for _ in 0..cpus {
        let handle = std::thread::spawn(move || worker(entries_per_page));
        handles.push(handle);
    }

    let progress_bar_handle = std::thread::spawn(move || simu_show_progress());

    let mut exception_value = Vec::new();

    for handle in handles {
        let mut r = handle.join().unwrap();
        exception_value.append(&mut r);
    }
    progress_bar_handle.join().unwrap();
    exception_value.sort();

    // 写入
    // 如果文件不存在，则创建文件，如果文件存在，则清空文件
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(format!("exception_value_{}.txt", entries_per_page))
        .unwrap();
    for v in exception_value {
        file.write_all(format!("{:?}\n", v).as_bytes()).unwrap();
    }
    println!("Simulate finished!");
}

fn check_page_num(results: &Vec<BuddyCalculatorResult>, total_pages: usize) -> bool {
    let idx = |order: usize| -> usize { order - MIN_ORDER };
    let mut cnt_pages = 0;
    for i in MIN_ORDER..MAX_ORDER {
        let r = &results[idx(i)];
        cnt_pages += r.pages;
        cnt_pages += r.entries * (1 << idx(i));
    }
    println!("cnt_pages: {}, total_pages: {}", cnt_pages, total_pages);
    if (cnt_pages as i64 - total_pages as i64).abs() > 1 {
        return false;
    }
    return true;
}

fn log2(x: usize) -> usize {
    let leading_zeros = x.leading_zeros() as usize;
    let log2x = 63 - leading_zeros;
    return log2x;
}
fn main() {
    // 读取参数
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <entries_per_page>", args[0]);
        return;
    }
    let entries_per_page: usize = args[1].parse().unwrap();
    // 单次计算
    // let total_memory = (19) * 4096;
    // let total_pages = total_memory / BuddyPageCalculator::PAGE_4K;
    // let total_pages = 67108864;
    // let mut calculator = BuddyPageCalculator::new(entries_per_page);
    // let r = calculator.calculate(total_pages);
    // print_results(&r.as_ref().unwrap());
    // calculator.check_result(&r.as_ref().unwrap()).unwrap();

    // 枚举所有的内存大小,由于最小需要19个页,因此小于19个页的错误都不用管
    simulate(entries_per_page);
}
