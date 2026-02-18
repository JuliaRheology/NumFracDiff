using NumFracDiff
using Documenter

DocMeta.setdocmeta!(NumFracDiff, :DocTestSetup, :(using NumFracDiff); recursive=true)

makedocs(;
    modules=[NumFracDiff],
    authors="Lorenzo Fonnesu, Andrea Grassi, Alessandra Bonfanti, Alexandre Kabla",
    sitename="NumFracDiff.jl",
    format=Documenter.HTML(;
        canonical="https://alebonfanti.github.io/NumFracDiff.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alebonfanti/NumFracDiff.jl",
    devbranch="main",
)
