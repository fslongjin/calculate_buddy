use std::{
    cmp,
    fs::OpenOptions,
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
};

use indicatif::{ProgressBar, ProgressStyle};

// 一个全局变量MAX_ORDER，用来表示buddy算法的最大阶数 [MIN_ORDER, MAX_ORDER)左闭右开区间
const MAX_ORDER: usize = 31;
// 4KB
const MIN_ORDER: usize = 12;

// 一个全局变量MAX_PAGES，用来表示buddy算法最多能够管理的页数
// const MAX_PAGES: usize = (1<<30);
const MAX_PAGES: usize = (1 << 48) / (1 << 12);

static CURRENT_PAGE: AtomicUsize = AtomicUsize::new(0);

// == 计算buddy所需要预分配的页数的代码 BEGIN ==

struct BuddyPageCalculator {
    layers: [BuddyCalculatorLayer; MAX_ORDER - MIN_ORDER],
    /// 总的页数
    total: usize,
    /// 每个页能够存放的buddy entry的数量
    entries_per_page: usize,
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
            total: 0,
            entries_per_page,
        }
    }

    fn calculate(&mut self, total_pages: usize) -> Vec<BuddyCalculatorResult> {
        self.total = total_pages;
        self.init();

        // 自上而下计算每一层需要的页数
        for start_order in (MIN_ORDER + 1..MAX_ORDER).rev() {
            self.calculate_layer(start_order);
        }

        let mut result: Vec<BuddyCalculatorResult> = Vec::new();

        for i in MIN_ORDER..MAX_ORDER {
            let l = &self.layers[i - MIN_ORDER];
            let ent = l.entries;
            let pages = (ent + self.entries_per_page - 1) / self.entries_per_page;
            result.push(BuddyCalculatorResult::new(i, pages, ent));
        }

        return result;
    }

    fn calculate_layer(&mut self, start_order: usize) {
        {
            let need_pages = calculator_layer!(self, start_order).need_pages();
            let got = calculator_layer!(self, MIN_ORDER).get_entries(need_pages);

            calculator_layer!(self, start_order).give_pages(got);
        }

        if calculator_layer!(self, start_order).need_pages() == 0 {
            return;
        }
        while calculator_layer!(self, start_order).need_pages() != 0 {
            // 4K entry不够，逐层向上找
            let mut ord: Option<usize> = None;
            for order in MIN_ORDER..(start_order + 1) {
                if calculator_layer!(self, order).free_entries() == 0 {
                    continue;
                }
                ord = Some(order);
                break;
            }

            assert!(ord.is_some());

            let order = ord.unwrap();
            // 逐层向下分裂
            for i in (MIN_ORDER + 1..order + 1).rev() {
                calculator_layer!(self, i).split();
                calculator_layer!(self, i - 1).add_entries(2);
            }
            // 把最底层的4K entry分配出去
            let need_pages = calculator_layer!(self, start_order).need_pages();
            let got = calculator_layer!(self, MIN_ORDER).get_entries(need_pages);
            calculator_layer!(self, start_order).give_pages(got);
        }
    }

    fn init(&mut self) {
        for i in (MIN_ORDER..MAX_ORDER).rev() {
            calculator_layer!(self, i).init_layer(self.entries_per_page);
        }

        self.init_max_layer();
    }

    fn init_max_layer(&mut self) {
        let l: &mut BuddyCalculatorLayer = &mut calculator_layer!(self, MAX_ORDER - 1);
        let ent = (self.total * Self::PAGE_4K) / Self::MAX_ORDER_SIZE;
        // println!(
        //     "total:{}, max_order_size:{}, ent:{ent}",
        //     self.total,
        //     Self::MAX_ORDER_SIZE
        // );
        l.add_entries(ent);
    }
}

#[derive(Debug, Clone, Copy)]
struct BuddyCalculatorLayer {
    /// 该层请求分配的页数（还没分配）
    need_pages: usize,
    /// 每个页能够存放的buddy entry的数量
    entries_per_page: usize,
    /// 当前层的buddy entry的数量
    entries: usize,
    /// 当前层剩余的插槽的数量
    slots_remain: usize,
}

impl BuddyCalculatorLayer {
    const fn new() -> Self {
        BuddyCalculatorLayer {
            need_pages: 0,
            entries_per_page: 0,
            entries: 0,
            slots_remain: 0,
        }
    }

    fn init_layer(&mut self, entries_per_page: usize) {
        self.entries_per_page = entries_per_page;
        self.need_pages = 0;
        self.entries = 0;
        self.slots_remain = 0;
    }

    fn add_entries(&mut self, mut entries: usize) {
        self.entries += entries;
        if self.slots_remain >= entries {
            self.slots_remain -= entries;
            return;
        }
        entries -= self.slots_remain;
        self.slots_remain = 0;

        let add_pages = (entries + self.entries_per_page - 1) / self.entries_per_page;
        self.need_pages += add_pages;
        self.slots_remain += self.entries_per_page * add_pages;
    }

    fn give_pages(&mut self, pages: usize) {
        // println!("self.need_pages: {}, pages: {}", self.need_pages, pages);
        assert!(self.need_pages >= pages);
        self.need_pages -= pages;
    }

    fn need_pages(&self) -> usize {
        self.need_pages
    }

    fn free_entries(&self) -> usize {
        self.entries
    }

    fn get_entries(&mut self, entries: usize) -> usize {
        let entries = cmp::min(entries, self.entries);
        self.entries -= entries;
        self.slots_remain += entries;

        self.reduce_page();
        return entries;
    }

    fn reduce_page(&mut self) {
        if self.slots_remain >= self.entries_per_page {
            let to_reduce = self.slots_remain / self.entries_per_page;
            self.need_pages -= to_reduce;
            self.slots_remain -= self.entries_per_page * to_reduce;
        }
    }

    fn split(&mut self) {
        assert!(self.entries > 0);
        self.entries -= 1;
        self.slots_remain += 1;
        self.reduce_page();
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

fn print_results(results: Vec<BuddyCalculatorResult>) {
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

fn check_ok(results: &Vec<BuddyCalculatorResult>) -> bool {
    let mut flag = false;
    for r in results.iter().rev() {
        if flag == false {
            if r.entries == 0 {
                continue;
            }
            flag = true;

            continue;
        }

        if r.pages > 1 {
            return false;
        }
    }

    return true;
}
fn worker() -> Vec<usize> {
    let mut exception_value = Vec::new();
    loop {
        let task: Option<Task> = task_generator();
        if task.is_none() {
            break;
        }

        let task = task.unwrap();

        for i in task.start_page..task.start_page + task.num_pages {
            let page = task.start_page + i;
            let mut calculator = BuddyPageCalculator::new(255);
            let r = calculator.calculate(page);
            if check_ok(&r) == false {
                exception_value.push(i);
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

fn simulate() {

    // 获取cpu核心数
    let cpus = num_cpus::get();

    let mut handles = Vec::new();
    for _ in 0..cpus {
        let handle = std::thread::spawn(move || worker());
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
        .open("exception_value.txt")
        .unwrap();
    for v in exception_value {
        file.write_all(format!("{}\n", v).as_bytes()).unwrap();
    }
    println!("Simulate finished!");
}

fn main() {
    // 单次计算
    // let total_memory = 1 << 52;
    // let total_pages = total_memory / BuddyPageCalculator::PAGE_4K;
    // let mut calculator = BuddyPageCalculator::new(255);
    // let r = calculator.calculate(total_pages);
    // print_results(r);

    // 枚举所有的内存大小
    simulate();
}
