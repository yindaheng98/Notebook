#include <stdio.h>

#define OK               0

  /* status code for successful run */

#define usage_error      1

  /* status code for improper syntax */

#define cannot_open_file 2

  /* status code for file access error */

#define READ_ONLY 0

  /* read access code for system open */

#define buf_size BUFSIZ

  /* stdio.h BUFSIZ chosen for efficiency */

#define print_count(n) printf("%8ld", n)

int status = OK;

  /* exit status of command, initially OK */

char *prog_name;

  /* who we are */

long tot_word_count, tot_line_count, 

     tot_char_count;

  /* total number of words, lines, chars */

wc_print(which, char_count, word_count, line_count)

  char *which;  /* which counts to print */

  long char_count, word_count, line_count;

    /* given totals */

{

  while (*which)

    switch (*which++) {

      case 'l': print_count(line_count);

        break;

      case 'w': print_count(word_count);

        break;

      case 'c': print_count(char_count);

        break;

      default:

        if ((status & usage_error) == 0) {

          fprintf(stderr,

            "\nUsage: %s [-lwc] [filename ...]\n",

            prog_name);

 	  status |= usage_error;

        }

    }

}

main(`argc, `argv)

  int argc;

    /* number of arguments on UNIX command line */

  char **argv;

    /* the arguments, an array of strings */

{

  int file_count;

    /* how many files there are */

  char *which;

    /* which counts to print */

  int `fd = 0;

    /* file descriptor, initialized to stdin */

  char buffer[buf_size];

    /* we read the input into this array */

  register char *ptr;

    /* first unprocessed character in buffer */

  register char *buf_end;

    /* the first unused position in buffer */

  register int c;

    /* current char, or # of chars just read */

  int in_word;

    /* are we within a word? */

  long word_count, line_count, char_count;

    /* # of words, lines, and chars so far */

  prog_name = argv[0];

  which = "lwc";

    /* if no option is given, print 3 values */

  if (argc > 1 && *argv[1] == '-') {

    which = argv[1] + 1;

    argc--;

    argv++;

  }

  file_count = argc - 1;

  argc--;

  do {

    if (file_count > 0

    && (fd = open(*(++argv), READ_ONLY)) < 0) {

      fprintf(stderr, 

        "%s: cannot open file %s\n",

        prog_name, *argv);

      status |= cannot_open_file;

      file_count--;

      continue;

    }

    ptr = buf_end = buffer;

    line_count = word_count = char_count = 0;

    in_word = 0;

    while (1) {

      if (ptr >= buf_end) {

        ptr = buffer;

        c = read(fd, ptr, buf_size);

        if (c <= 0) break;

        char_count += c;

        buf_end = buffer + c;

      }

      c = *ptr++;

      if (c > ' ' && c < 0177) {

        /* visible ASCII codes */

        if (!in_word) {

          word_count++;

          in_word = 1;

        }

        continue;

      }

      if (c == '\n') line_count++;

      else if (c != ' ' && c != '\t') continue;

      in_word = 0;

        /* c is newline, space, or tab */

    }

    wc_print(which, char_count, word_count, 

             line_count);

    if (file_count)

      printf(" %s\n", *argv); /* not stdin */

    else 

      printf("\n");           /* stdin */

    close(fd);

    tot_line_count += line_count;

    tot_word_count += word_count;

    tot_char_count += char_count;

      /* even if there is only one file */

  } while (--argc > 0);

  if (file_count > 1) {

    wc_print(which, tot_char_count, 

             tot_word_count, tot_line_count);

    printf(" total in %d files\n", file_count);

  }

  exit(status);

}

