# Linux系统的启动过程

## 总括

| 步骤 | 部件 | 全称 | 作用 |
|:---:|:---:|:---|:---|
| 1 | BIOS | Basic I/O System | executes MBR|
| 2 | MBR | Master Boot Record | executes GRUB|
| 3 | GRUB | Grand Unified Bootloader | executes Kernel|
| 4 | Kernel | Kernel | executes /sbin/init|
| 5 | Init | Init | executes runlevel programs|
| 6 | Runlevel | Runlevel | Runlevel programs are executed from /etc/rc.d/rc*.d|
