# LATEX生态中与文学编程思想有关的工具：doc和docStrip包

LATEX生态系统中的doc/docStrip包虽不及ipython jupyter的设计精妙复杂且用途广泛，但仍是文学编程思想在LATEX生态中的一种体现。它赋予了TEX文学编程的能力。正如doc包的作者在其文档开头所写：

>The TEX macros which are described here allow definitions and documentation to be held in one and the same file. This has the advantage that normally very complicated instructions are made simpler to understand by comments inside the definition. In addition to this, updates are easier and only one source file needs to be changed. 

文学编程这种寓代码于文档中的方式会使得编程维护变得更加简单。然而，也正如docStrip的作者在docStrip的文档开头所写：

>This way of writing TEX programs obviously has great advantages, especially when the program becomes larger than a couple of macros. There is one drawback however, and that is that such programs may take longer than expected to run because TEX is an interpreter and has to decide for each line of the program file what it has to do with it. Therefore, TEX programs may be sped up by removing all comments from them. 

非结构化的文学编程方式在性能上的弊端也非常明显。这也是作者创造docStrip包的原因。用于自动将TEX包文档的代码删去，只保留TEX代码，从而在后续操作中提高性能，这一过程完全可以看作是文学编程中的Tangle过程。
