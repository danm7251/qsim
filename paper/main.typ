#set page(
    paper: "a4",
    margin: 2.5cm,
)
#set text(font: "Arial")

#align(center)[
    #include "chapters/title.typ"  
]
#pagebreak()

#set heading(numbering: "1.")
#outline()
#pagebreak()

#include "chapters/abstract.typ"
#pagebreak()

#include "chapters/introduction.typ"
#pagebreak()

#include "chapters/background.typ"
#pagebreak()

#include "chapters/design.typ"
#pagebreak()

#include "chapters/implementation.typ"
#pagebreak()

#include "chapters/evaluation.typ"
#pagebreak()

#include "chapters/discussion.typ"
#pagebreak()

#include "chapters/conclusion.typ"
#pagebreak()

#include "chapters/references.typ"