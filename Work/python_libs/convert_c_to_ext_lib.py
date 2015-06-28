from __future__ import print_function    # Python 3 compatibility

import os
from textwrap import dedent
import numpy as np
from scipy.weave import ext_tools



def parse_c_file(filename):
    """
    Parses a C/C++ file "filename" (with special formatting, see "c2py_example/test.c" for an example),
    and returns a tuple containing: (module_name, libraries, includes, supportcode, functions),
    useful for use in "scipy.weave.ext_tools".
    
    The module's name will be "<filename without extension>_ext".
    """

    # Define extension module name
    module_name = os.path.splitext(os.path.basename(filename))[0] + "_ext"
    
    # Open file
    lines = open(filename, 'r').read().split('\n')
    
    # Read sections
    libs_start                          = lines.index("/* Libraries */")
    includes_start           = libs_end = lines.index("/* Includes */")
    supportcode_start    = includes_end = lines.index("/* Support code */")
    funcs_start       = supportcode_end = lines.index("/* Functions exported to Python */")
    funcs_end                           = len(lines)
    
    # Parse "Libraries" section
    libraries = []
    for line in lines[libs_start+1 : libs_end]:
        start = line.find("/*")
        end = line.find("*/", start + 2)
        if start != -1 != end:
            libraries.append(line[start+2 : end].strip())
    
    # Parse "Includes" section
    includes = []
    for line in lines[includes_start+1 : includes_end]:
        line = line.strip()
        if line.startswith("#include"):
            includes.append(line[len("#include"):].strip())
    
    # Parse "Support code" section
    supportcode = '\n'.join(lines[supportcode_start+1 : supportcode_end]) + '\n'
    
    # Parse "Functions exported to Python" section
    functions = []    # will contain dicts with the following keys: "name", "code", and ("doc", if present), and ("arg_names" and "arg_instances", if arguments present)
    function = {}
    linenr = funcs_start + 1
    while linenr < funcs_end:
        line = lines[linenr]
        linenr += 1
        
        if not "name" in function:
            if not "doc" in function:    # in this case we seek for the beginning of a C comments section (used as doc)
                if line.strip().startswith("/*"):
                    try:
                        code_start = lines.index('{', linenr)
                    except ValueError:
                        break
                    function["doc"] = '\n'.join(lines[linenr-1 : code_start-1])
            
            if not "arg_names" in function:    # in this case we seek for the beginning of an arguments section
                if line.find("* Arguments:") != -1:
                    function["arg_names"] = []
                    function["arg_instances"] = []
                    continue
            
            else:    # in this case we seek for arguments
                start = line.find("* ")    # start/continue of C comment
                end = line.find("#", start)    # start of Python comment
                if end == -1:
                    end = len(line)
                mid = line.find("=", start, end)    # middle of Python equality
                if start != -1 != mid:
                    function["arg_names"].append(line[start+2 : mid].strip())
                    function["arg_instances"].append(eval(line[mid+1 : end].strip()))
                    continue
            
            # If no doc or arguments found on this line, we seek for the function name
            if line.startswith("void ") and (line.endswith("()") or line.endswith("(/* ... */)")):
                function["name"] = line[len("void ") : line.find("(")]
        
        elif not "code" in function:    # in this case we seek for the function code
            if line == '{':
                try:
                    stop = lines.index('}', linenr)
                except ValueError:
                    break
                function["code"] = '\n'.join(lines[linenr : stop]) + '\n'
                functions.append(function)
                function = {}
                linenr = stop + 1
    
    # Return results
    return module_name, libraries, includes, supportcode, functions


def create_ext_lib(filename, openmp=False):
    """
    Create an extension module library from the
    given C/C++ file "filename" (with special formatting, see "c2py_example/test.c" for an example),
    and returns the new extension module object.
    
    "openmp" : if False, openmp pragmas will be disabled
    """

    # Read extension module definition by C file
    module_name, libraries, includes, supportcode, functions = parse_c_file(filename)
    #print ("Libraries:")
    #print (libraries)
    #print ()
    #print ("Includes:")
    #print (includes)
    #print ()
    #print ("Support code:")
    #print (supportcode)
    #print ()
    #print ("Functions:")
    #for function in functions:
        #print (function)
    #print ()

    # Create a new extension module object
    module = ext_tools.ext_module(module_name)

    # Add the libraries
    for library in libraries:
        module.customize.add_library(library)

    # Add the headers
    for header in includes:
        module.customize.add_header(header)

    if openmp:
        # Enable the use of parallel functions
        module.customize.set_compiler("gcc")
        module.customize.add_extra_compile_arg("-fopenmp")
        module.customize.add_extra_link_arg("-fopenmp")
    else:
        # Comment openmp pragmas
        supportcode = supportcode.replace("#pragma omp ", "//#pragma omp ")
        for function in functions:
            function["code"] = function["code"].replace("#pragma omp ", "//#pragma omp ")

    # Add the support-code
    module.customize.add_support_code(supportcode)

    # Add the module's functions

    def add_function(_module, _func):
        _arg_names = []
        if "arg_names" in _func:
            _arg_names = _func["arg_names"]
            intersection = set(locals()) & set(_func["arg_names"])
            if intersection:
                raise ValueError("The following argument-names collide with Python locals(): %s" % list(intersection))
            locals().update(dict(zip(_arg_names, _func["arg_instances"])))
        
        _module.add_function(ext_tools.ext_function(_func["name"], _func["code"], _arg_names))

    for function in functions:
        add_function(module, function)

    # Create the functions' doc-strings, a little bit hacky since weave doesn't really support it natively

    forward_decls = ''.join([
            "static PyObject* %s(PyObject*self, PyObject* args, PyObject* kywds);\n" % function["name"]
            for function in functions ])
    functions_struct = \
            """
            static PyMethodDef compiled_methods_with_docs[] = 
            {
                %s
                {NULL,      NULL}        /* Sentinel */
            };
            """
    function_defs = [
            '{"%s", (PyCFunction)%s, METH_VARARGS|METH_KEYWORDS, "%s"},' %
            (function["name"], function["name"], (
                    function["doc"].replace('\\', "\\\\").replace('"', '\\"').replace('\n', "\\n") if "doc" in function
                    else "" ))
            for function in functions ]
    functions_struct = forward_decls + (dedent(functions_struct) % "\n    ".join(function_defs))

    init_function_struct = '(void) Py_InitModule("%s", compiled_methods_with_docs); \n//' % module_name

    module.customize.add_support_code(functions_struct)
    module.customize.add_module_init_code(init_function_struct)

    # (Re-)Compile the extension module
    print ('Compiling "%s" to extension module "%s"...' % (filename, module_name))
    print ()
    for f in os.listdir('.'):
        filename, extension = os.path.splitext(f)
        if filename == module_name and extension in (".cpp", ".so", ".dylib", ".dll") and os.path.isfile(f):
            os.remove(f)    # remove cpp and library, to force recompilation
    module.compile()
    print ("Done.")
    print ()
    
    return module
