# (未完成)Linux中的`sed`指令

```
yin@DESKTOP-IG564I6:/mnt/e/MyPrograms/My-docs$ sed
Usage: sed [OPTION]... {script-only-if-no-other-script} [input-file]...

  -n, --quiet, --silent
                 suppress automatic printing of pattern space
  -e script, --expression=script
                 add the script to the commands to be executed
  -f script-file, --file=script-file
                 add the contents of script-file to the commands to be executed
  --follow-symlinks
                 follow symlinks when processing in place
  -i[SUFFIX], --in-place[=SUFFIX]
                 edit files in place (makes backup if SUFFIX supplied)
  -l N, --line-length=N
                 specify the desired line-wrap length for the `l' command
  --posix
                 disable all GNU extensions.
  -E, -r, --regexp-extended
                 use extended regular expressions in the script
                 (for portability use POSIX -E).
  -s, --separate
                 consider files as separate rather than as a single,
                 continuous long stream.
      --sandbox
                 operate in sandbox mode.
  -u, --unbuffered
                 load minimal amounts of data from the input files and flush
                 the output buffers more often
  -z, --null-data
                 separate lines by NUL characters
      --help     display this help and exit
      --version  output version information and exit

If no -e, --expression, -f, or --file option is given, then the first
non-option argument is taken as the sed script to interpret.  All
remaining arguments are names of input files; if no input files are
specified, then the standard input is read.

GNU sed home page: <http://www.gnu.org/software/sed/>.
General help using GNU software: <http://www.gnu.org/gethelp/>.
```

## `sed`部分参数解析

* -n, --quiet, --silent：使用安静(silent)模式。在一般 sed 的用法中，所有来自 STDIN 的数据一般都会被列出到终端上。但如果加上 -n 参数后，则只有经过sed 特殊处理的那一行(或者动作)才会被列出来。
* -e script, --expression=script：直接在命令行模式上进行sed动作编辑，此为默认选项
* -f script-file, --file=script-file：将sed的动作写在一个文件内，用–f filename 执行filename内的sed动作
* -E, -r, --regexp-extended：支持扩展正则表达式
* -i[SUFFIX], --in-place[=SUFFIX]：sed动作直接修改文件内容

## `sed`动作命令解析

### 查询动作

动作指令|对应动作
-|-
x|x为行号，查询第x行
x,y|表示行号从x到y，查询第x行到第y行
/pattern|查询包含模式的行
/pattern /pattern|查询包含两个模式的行
pattern/,x|在给定行号上查询包含模式的行
x,/pattern/|通过行号和模式查询匹配的行
x,y!|查询不包含指定行号x和y的行