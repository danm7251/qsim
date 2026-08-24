#import "@preview/wordometer:0.1.5": word-count, total-words
#show: word-count.with(
    exclude: (
        <no-wc>,
        heading,
    )
)

#let submission = false

#let note(body) = if submission {
  panic("Submission build contains working notes!")
} else {
  block(
    fill: luma(85%),
    inset: 8pt,
    body,
  )
}

#set page(
    paper: "a4",
    margin: 2.5cm,
)
#set text(font: "Arial")

#[
    #text(size: 14pt)[
        Word count: #total-words / 14k
    ]

    #align(center)[
        #include "chapters/title.typ"  
    ]
    #pagebreak()

    #set heading(numbering: "1.")
    #outline()
    #pagebreak()
] <no-wc>

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

#[
    #include "chapters/references.typ"
] <no-wc>