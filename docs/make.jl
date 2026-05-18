using NumFracDiff
using Documenter
using Literate

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
                    repo_root_url = "github.com/JuliaRheology/NumFracDiff", 

                )
        end
    end
end 

function maindocbuilder()
    docprepare()

    makedocs(;
        modules=[NumFracDiff],
        authors="Lorenzo Fonnesu, Andrea Grassi, Alessandra Bonfanti, Alexandre Kabla",
        repo="github.com/JuliaRheology/NumFracDiff",
        sitename="NumFracDiff.jl",
        source="staging-docs",
        format=Documenter.HTML(),#;
        #     canonical="https://alebonfanti.github.io/NumFracDiff.jl",
        #     edit_link="main",
        #     assets=String[],
        # ),
        pages=[
            "Home" => "index.md",
            "Architecture" => "Architecture.md",
            "Numerical Methods" => ["GL" => "gl.md",
                                    "RL" => "rl.md",
                                    "Caputo" => "caputo.md"],
            "Usage" => "usage.md",
            "API" => "API.md"
        ],
    )

    deploydocs(;
        repo="github.com/JuliaRheology/NumFracDiff.git",
        devbranch="Documentation",
        deps = nothing,
        make = nothing,
        target = "build"
    )
end

maindocbuilder()