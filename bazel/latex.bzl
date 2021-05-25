def _latex_pdf_impl(ctx):
    root = ctx.build_file_path[:-5]
    srcs = ctx.files.srcs
    cmd_tpl = "latexmk -pdf -outdir={} {}"
    cmds = []
    outs = []
    for x in srcs:
        out = x.path[len(root):].replace(".tex", ".pdf")
        out = ctx.actions.declare_file(out)
        cmds.append(cmd_tpl.format(out.dirname, x.path))
        outs.append(out)
    ctx.actions.run_shell(
        inputs = srcs + ctx.files.data + ctx.files.deps,
        outputs = outs,
        use_default_shell_env = True,
        progress_message = " && ".join(cmds),
        command = " && ".join(cmds),
    )
    return [DefaultInfo(files = depset(outs))]

_latex_pdf = rule(
    implementation = _latex_pdf_impl,
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = [".tex"],
        ),
        "deps": attr.label_list(
            allow_empty = True,
            allow_files = [".tex", ".cls", ".sty"],
            default = [],
        ),
        "data": attr.label_list(
            allow_empty = True,
            allow_files = True,
            default = [],
        ),
        "opt": attr.string_list(
            allow_empty = True,
            default = [],
        ),
    },
)

script_template = """\
#!/bin/sh

openpdf () {{
    if type xdg-open > /dev/null 2>&1; then # X11 Linux
        exec xdg-open $1 2>/dev/null &
    elif type open > /dev/null 2>&1; then # macOS
        exec open $1
    else
        echo "Don't know how to view PDFs on this platform." >&2
        exit 1
    fi
}}

if [ $# -ne 0 ];
then
    for pdf in $@
    do
        openpdf {root}$pdf
    done
else
    for pdf in "{pdfs}"
    do
        openpdf $pdf
    done
fi
"""

def _pdf_view_impl(ctx):
    root = ctx.build_file_path[:-5]
    pdfs = []
    for x in ctx.files.pdfs:
        pdfs.append(x.path[len(ctx.bin_dir.path) + 1:])

    script = ctx.actions.declare_file(ctx.label.name)
    script_content = script_template.format(
        pdfs = " ".join(pdfs),
        root = root,
    )
    ctx.actions.write(script, script_content, is_executable = True)

    runfiles = ctx.runfiles(files = ctx.files.pdfs)
    return [DefaultInfo(executable = script, runfiles = runfiles)]

_pdf_view = rule(
    implementation = _pdf_view_impl,
    executable = True,
    attrs = {
        "pdfs": attr.label_list(
            mandatory = True,
            allow_files = [".pdf"],
        ),
    },
)

def latex_pdf(name, srcs, deps = [], data = [], opt = [], tags = []):
    _latex_pdf(
        name = name + "_pdf",
        srcs = srcs,
        deps = deps,
        data = data,
        opt = opt,
        tags = ["latex"] + tags,
    )

    _pdf_view(
        name = name,
        pdfs = [name + "_pdf"],
        tags = ["latex"] + tags,
    )

def latex_library(name, srcs, **kwargs):
    native.filegroup(
        name = name,
        srcs = srcs,
        tags = ["latex"],
        **kwargs
    )
