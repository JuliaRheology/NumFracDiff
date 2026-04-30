using NumFracDiff
using Documenter

DocMeta.setdocmeta!(NumFracDiff, :DocTestSetup, :(using NumFracDiff); recursive=true)


function docprepare()
    # remove remnants of previous build and recreate staging dir
    rm("docs/staging-docs", force=true, recursive=true)
    mkdir("docs/staging-docs")

    # copy readme to staging-docs, remove Logo image
    # write("docs/staging-docs/index.md", 
    #         open("README.md") do input
    #             # readuntil(input, "<!-- delim -->", keep = true)
    #             read(input)
    #         end)
    
    # copy assets to staging directory
    # cp("docs/src/assets", "docs/staging-docs/assets")

    # iterate through src and convert/copy as appropriate
    for file in readdir("docs/src")
        if endswith(file, "md")
            cp("docs/src/$file", "docs/staging-docs/$file")
        elseif endswith(file, "jl")
                Literate.markdown(
                    "docs/src/$file",
                    "docs/staging-docs/";
                    documenter = true,
                    repo_root_url = "https://github.com/JuliaRheology/RHEOS.jl", 

                )
        end
    end
end 

function maindocbuilder()
    docprepare()

    makedocs(;
        modules=[NumFracDiff],
        authors="Lorenzo Fonnesu, Andrea Grassi, Alessandra Bonfanti, Alexandre Kabla",
        repo="https://github.com/JuliaRheology/NumFracDiff",
        sitename="NumFracDiff.jl",
        source="staging-docs",
        format=Documenter.HTML(),#;
        #     canonical="https://alebonfanti.github.io/NumFracDiff.jl",
        #     edit_link="main",
        #     assets=String[],
        # ),
        pages=[
            "Home" => "index.md",
        ],
    )

    deploydocs(;
        repo="https://github.com/JuliaRheology/NumFracDiff",
        devbranch="Documentation",
        deps = nothing,
        make = nothing,
        target = "build"
    )
end

maindocbuilder()