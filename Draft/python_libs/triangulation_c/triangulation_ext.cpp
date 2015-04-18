#ifdef __CPLUSPLUS__
extern "C" {
#endif

#ifndef __GNUC__
#pragma warning(disable: 4275)
#pragma warning(disable: 4101)

#endif
#include "Python.h"
#include "compile.h"
#include "frameobject.h"
#include <complex>
#include <math.h>
#include <string>
#include "scxx/object.h"
#include "scxx/list.h"
#include "scxx/tuple.h"
#include "scxx/dict.h"
#include <iostream>
#include <stdio.h>
#include "numpy/arrayobject.h"
#include <opencv2/imgproc/imgproc.hpp>




// global None value for use in functions.
namespace py {
object None = object(Py_None);
}

const char* find_type(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    if(PyFile_Check(py_obj)) return "file";
    if(PyModule_Check(py_obj)) return "module";

    //should probably do more intergation (and thinking) on these.
    if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance";
    if(PyCallable_Check(py_obj)) return "callable";
    return "unknown type";
}

void throw_error(PyObject* exc, const char* msg)
{
 //printf("setting python error: %s\n",msg);
  PyErr_SetString(exc, msg);
  //printf("throwing error\n");
  throw 1;
}

void handle_bad_type(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}

void handle_conversion_error(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}


class int_handler
{
public:
    int convert_to_int(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInt_Check(py_obj))
            handle_conversion_error(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }

    int py_to_int(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInt_Check(py_obj))
            handle_bad_type(py_obj,"int", name);
        
        return (int) PyInt_AsLong(py_obj);
    }
};

int_handler x__int_handler = int_handler();
#define convert_to_int(py_obj,name) \
        x__int_handler.convert_to_int(py_obj,name)
#define py_to_int(py_obj,name) \
        x__int_handler.py_to_int(py_obj,name)


PyObject* int_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class float_handler
{
public:
    double convert_to_float(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_conversion_error(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }

    double py_to_float(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_bad_type(py_obj,"float", name);
        
        return PyFloat_AsDouble(py_obj);
    }
};

float_handler x__float_handler = float_handler();
#define convert_to_float(py_obj,name) \
        x__float_handler.convert_to_float(py_obj,name)
#define py_to_float(py_obj,name) \
        x__float_handler.py_to_float(py_obj,name)


PyObject* float_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class complex_handler
{
public:
    std::complex<double> convert_to_complex(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_conversion_error(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }

    std::complex<double> py_to_complex(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_bad_type(py_obj,"complex", name);
        
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }
};

complex_handler x__complex_handler = complex_handler();
#define convert_to_complex(py_obj,name) \
        x__complex_handler.convert_to_complex(py_obj,name)
#define py_to_complex(py_obj,name) \
        x__complex_handler.py_to_complex(py_obj,name)


PyObject* complex_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class unicode_handler
{
public:
    Py_UNICODE* convert_to_unicode(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_conversion_error(py_obj,"unicode", name);
        return PyUnicode_AS_UNICODE(py_obj);
    }

    Py_UNICODE* py_to_unicode(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_bad_type(py_obj,"unicode", name);
        Py_XINCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);
    }
};

unicode_handler x__unicode_handler = unicode_handler();
#define convert_to_unicode(py_obj,name) \
        x__unicode_handler.convert_to_unicode(py_obj,name)
#define py_to_unicode(py_obj,name) \
        x__unicode_handler.py_to_unicode(py_obj,name)


PyObject* unicode_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class string_handler
{
public:
    std::string convert_to_string(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return std::string(PyString_AsString(py_obj));
    }

    std::string py_to_string(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyString_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        Py_XINCREF(py_obj);
        return std::string(PyString_AsString(py_obj));
    }
};

string_handler x__string_handler = string_handler();
#define convert_to_string(py_obj,name) \
        x__string_handler.convert_to_string(py_obj,name)
#define py_to_string(py_obj,name) \
        x__string_handler.py_to_string(py_obj,name)


               PyObject* string_to_py(std::string s)
               {
                   return PyString_FromString(s.c_str());
               }
               
class list_handler
{
public:
    py::list convert_to_list(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return py::list(py_obj);
    }

    py::list py_to_list(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyList_Check(py_obj))
            handle_bad_type(py_obj,"list", name);
        
        return py::list(py_obj);
    }
};

list_handler x__list_handler = list_handler();
#define convert_to_list(py_obj,name) \
        x__list_handler.convert_to_list(py_obj,name)
#define py_to_list(py_obj,name) \
        x__list_handler.py_to_list(py_obj,name)


PyObject* list_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class dict_handler
{
public:
    py::dict convert_to_dict(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return py::dict(py_obj);
    }

    py::dict py_to_dict(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyDict_Check(py_obj))
            handle_bad_type(py_obj,"dict", name);
        
        return py::dict(py_obj);
    }
};

dict_handler x__dict_handler = dict_handler();
#define convert_to_dict(py_obj,name) \
        x__dict_handler.convert_to_dict(py_obj,name)
#define py_to_dict(py_obj,name) \
        x__dict_handler.py_to_dict(py_obj,name)


PyObject* dict_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class tuple_handler
{
public:
    py::tuple convert_to_tuple(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return py::tuple(py_obj);
    }

    py::tuple py_to_tuple(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_bad_type(py_obj,"tuple", name);
        
        return py::tuple(py_obj);
    }
};

tuple_handler x__tuple_handler = tuple_handler();
#define convert_to_tuple(py_obj,name) \
        x__tuple_handler.convert_to_tuple(py_obj,name)
#define py_to_tuple(py_obj,name) \
        x__tuple_handler.py_to_tuple(py_obj,name)


PyObject* tuple_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class file_handler
{
public:
    FILE* convert_to_file(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"file", name);
        return PyFile_AsFile(py_obj);
    }

    FILE* py_to_file(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"file", name);
        Py_XINCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
};

file_handler x__file_handler = file_handler();
#define convert_to_file(py_obj,name) \
        x__file_handler.convert_to_file(py_obj,name)
#define py_to_file(py_obj,name) \
        x__file_handler.py_to_file(py_obj,name)


               PyObject* file_to_py(FILE* file, const char* name,
                                    const char* mode)
               {
                   return (PyObject*) PyFile_FromFile(file,
                     const_cast<char*>(name),
                     const_cast<char*>(mode), fclose);
               }
               
class instance_handler
{
public:
    py::object convert_to_instance(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_conversion_error(py_obj,"instance", name);
        return py::object(py_obj);
    }

    py::object py_to_instance(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_bad_type(py_obj,"instance", name);
        
        return py::object(py_obj);
    }
};

instance_handler x__instance_handler = instance_handler();
#define convert_to_instance(py_obj,name) \
        x__instance_handler.convert_to_instance(py_obj,name)
#define py_to_instance(py_obj,name) \
        x__instance_handler.py_to_instance(py_obj,name)


PyObject* instance_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class numpy_size_handler
{
public:
    void conversion_numpy_check_size(PyArrayObject* arr_obj, int Ndims,
                                     const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"Conversion Error: received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_size(PyArrayObject* arr_obj, int Ndims, const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_size_handler x__numpy_size_handler = numpy_size_handler();
#define conversion_numpy_check_size x__numpy_size_handler.conversion_numpy_check_size
#define numpy_check_size x__numpy_size_handler.numpy_check_size


class numpy_type_handler
{
public:
    void conversion_numpy_check_type(PyArrayObject* arr_obj, int numeric_type,
                                     const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {

        const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                "float", "double", "longdouble", "cfloat", "cdouble",
                                "clongdouble", "object", "string", "unicode", "void", "ntype",
                                "unknown"};
        char msg[500];
        sprintf(msg,"Conversion Error: received '%s' typed array instead of '%s' typed array for variable '%s'",
                type_names[arr_type],type_names[numeric_type],name);
        throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_type(PyArrayObject* arr_obj, int numeric_type, const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {
            const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                    "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                    "float", "double", "longdouble", "cfloat", "cdouble",
                                    "clongdouble", "object", "string", "unicode", "void", "ntype",
                                    "unknown"};
            char msg[500];
            sprintf(msg,"received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_type_handler x__numpy_type_handler = numpy_type_handler();
#define conversion_numpy_check_type x__numpy_type_handler.conversion_numpy_check_type
#define numpy_check_type x__numpy_type_handler.numpy_check_type


class numpy_handler
{
public:
    PyArrayObject* convert_to_numpy(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyArray_Check(py_obj))
            handle_conversion_error(py_obj,"numpy", name);
        return (PyArrayObject*) py_obj;
    }

    PyArrayObject* py_to_numpy(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyArray_Check(py_obj))
            handle_bad_type(py_obj,"numpy", name);
        Py_XINCREF(py_obj);
        return (PyArrayObject*) py_obj;
    }
};

numpy_handler x__numpy_handler = numpy_handler();
#define convert_to_numpy(py_obj,name) \
        x__numpy_handler.convert_to_numpy(py_obj,name)
#define py_to_numpy(py_obj,name) \
        x__numpy_handler.py_to_numpy(py_obj,name)


PyObject* numpy_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class catchall_handler
{
public:
    py::object convert_to_catchall(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !(py_obj))
            handle_conversion_error(py_obj,"catchall", name);
        return py::object(py_obj);
    }

    py::object py_to_catchall(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !(py_obj))
            handle_bad_type(py_obj,"catchall", name);
        
        return py::object(py_obj);
    }
};

catchall_handler x__catchall_handler = catchall_handler();
#define convert_to_catchall(py_obj,name) \
        x__catchall_handler.convert_to_catchall(py_obj,name)
#define py_to_catchall(py_obj,name) \
        x__catchall_handler.py_to_catchall(py_obj,name)


PyObject* catchall_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


#define P1_(k,l)    (P1[4 * (k) + (l)])
#define P2_(k,l)    (P2[4 * (k) + (l)])

#define u1_(j)    (u1[2 * xi + (j)])
#define u2_(j)    (u2[2 * xi + (j)])

#define construct_A()    double A[12] = { \
        u1_(0) * P1_(2,0) - P1_(0,0),    u1_(0) * P1_(2,1) - P1_(0,1),    u1_(0) * P1_(2,2) - P1_(0,2), \
        u1_(1) * P1_(2,0) - P1_(1,0),    u1_(1) * P1_(2,1) - P1_(1,1),    u1_(1) * P1_(2,2) - P1_(1,2), \
        u2_(0) * P2_(2,0) - P2_(0,0),    u2_(0) * P2_(2,1) - P2_(0,1),    u2_(0) * P2_(2,2) - P2_(0,2), \
        u2_(1) * P2_(2,0) - P2_(1,0),    u2_(1) * P2_(2,1) - P2_(1,1),    u2_(1) * P2_(2,2) - P2_(1,2) }

#define construct_b()    double b[4] = { \
        -(u1_(0) * P1_(2,3) - P1_(0,3)), \
        -(u1_(1) * P1_(2,3) - P1_(1,3)), \
        -(u2_(0) * P2_(2,3) - P2_(0,3)), \
        -(u2_(1) * P2_(2,3) - P2_(1,3)) }

#define x_(j)    (x[3 * xi + (j)])



static PyObject* linear_LS_triangulation(PyObject*self, PyObject* args, PyObject* kywds);
static PyObject* iterative_LS_triangulation(PyObject*self, PyObject* args, PyObject* kywds);

static PyMethodDef compiled_methods_with_docs[] = 
{
    {"linear_LS_triangulation", (PyCFunction)linear_LS_triangulation, METH_VARARGS|METH_KEYWORDS, "/* Arguments:\n *      u1 = np.empty((0, 2), dtype=np.float64)    # [in]\n *      P1 = np.empty((3, 4), dtype=np.float64)    # [in]\n *      u2 = np.empty((0, 2), dtype=np.float64)    # [in]\n *      P2 = np.empty((3, 4), dtype=np.float64)    # [in]\n * \n *       x = np.empty((0, 3), dtype=np.float64)    # [out]\n * \n *  Note 1:\n *      The dimensions with size 0 should be replaced by the amount of points to triangulate.\n *  Note 2:\n *      The data of each numpy array argument \"a\" should be aligned,\n *      to check whether this is the case, the following code shouldn't raise an exception:\n *      #    v = a.view(); v.shape = (a.size,)\n *      To fix this automatically, run:\n *      #    a = a.reshape(a.size).reshape(a.shape)\n */"},
    {"iterative_LS_triangulation", (PyCFunction)iterative_LS_triangulation, METH_VARARGS|METH_KEYWORDS, "/* Arguments:\n *             u1 = np.empty((0, 2), dtype=np.float64)     # [in]\n *             P1 = np.empty((3, 4), dtype=np.float64)     # [in]\n *             u2 = np.empty((0, 2), dtype=np.float64)     # [in]\n *             P2 = np.empty((3, 4), dtype=np.float64)     # [in]\n *      tolerance = float()                                # [in]\n * \n *             x = np.empty((0, 3), dtype=np.float64)    # [out]\n *      x_status = np.empty( 0    , dtype=np.int32)      # [out]\n * \n *  Note 1:\n *      The dimensions with size 0 should be replaced by the amount of points to triangulate.\n *  Note 2:\n *      The data of each numpy array argument \"a\" should be aligned,\n *      to check whether this is the case, the following code shouldn't raise an exception:\n *      #    v = a.view(); v.shape = (a.size,)\n *      To fix this automatically, run:\n *      #    a = a.reshape(a.size).reshape(a.shape)\n */"},
    {NULL,      NULL}        /* Sentinel */
};


static PyObject* linear_LS_triangulation(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"u1","P1","u2","P2","x","local_dict", NULL};
    PyObject *py_u1, *py_P1, *py_u2, *py_P2, *py_x;
    int u1_used, P1_used, u2_used, P2_used, x_used;
    py_u1 = py_P1 = py_u2 = py_P2 = py_x = NULL;
    u1_used= P1_used= u2_used= P2_used= x_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOO|O:linear_LS_triangulation",const_cast<char**>(kwlist),&py_u1, &py_P1, &py_u2, &py_P2, &py_x, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_u1 = py_u1;
        PyArrayObject* u1_array = convert_to_numpy(py_u1,"u1");
        conversion_numpy_check_type(u1_array,PyArray_DOUBLE,"u1");
        #define U11(i) (*((double*)(u1_array->data + (i)*Su1[0])))
        #define U12(i,j) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1])))
        #define U13(i,j,k) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1] + (k)*Su1[2])))
        #define U14(i,j,k,l) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1] + (k)*Su1[2] + (l)*Su1[3])))
        npy_intp* Nu1 = u1_array->dimensions;
        npy_intp* Su1 = u1_array->strides;
        int Du1 = u1_array->nd;
        double* u1 = (double*) u1_array->data;
        u1_used = 1;
        py_P1 = py_P1;
        PyArrayObject* P1_array = convert_to_numpy(py_P1,"P1");
        conversion_numpy_check_type(P1_array,PyArray_DOUBLE,"P1");
        #define P11(i) (*((double*)(P1_array->data + (i)*SP1[0])))
        #define P12(i,j) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1])))
        #define P13(i,j,k) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1] + (k)*SP1[2])))
        #define P14(i,j,k,l) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1] + (k)*SP1[2] + (l)*SP1[3])))
        npy_intp* NP1 = P1_array->dimensions;
        npy_intp* SP1 = P1_array->strides;
        int DP1 = P1_array->nd;
        double* P1 = (double*) P1_array->data;
        P1_used = 1;
        py_u2 = py_u2;
        PyArrayObject* u2_array = convert_to_numpy(py_u2,"u2");
        conversion_numpy_check_type(u2_array,PyArray_DOUBLE,"u2");
        #define U21(i) (*((double*)(u2_array->data + (i)*Su2[0])))
        #define U22(i,j) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1])))
        #define U23(i,j,k) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1] + (k)*Su2[2])))
        #define U24(i,j,k,l) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1] + (k)*Su2[2] + (l)*Su2[3])))
        npy_intp* Nu2 = u2_array->dimensions;
        npy_intp* Su2 = u2_array->strides;
        int Du2 = u2_array->nd;
        double* u2 = (double*) u2_array->data;
        u2_used = 1;
        py_P2 = py_P2;
        PyArrayObject* P2_array = convert_to_numpy(py_P2,"P2");
        conversion_numpy_check_type(P2_array,PyArray_DOUBLE,"P2");
        #define P21(i) (*((double*)(P2_array->data + (i)*SP2[0])))
        #define P22(i,j) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1])))
        #define P23(i,j,k) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1] + (k)*SP2[2])))
        #define P24(i,j,k,l) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1] + (k)*SP2[2] + (l)*SP2[3])))
        npy_intp* NP2 = P2_array->dimensions;
        npy_intp* SP2 = P2_array->strides;
        int DP2 = P2_array->nd;
        double* P2 = (double*) P2_array->data;
        P2_used = 1;
        py_x = py_x;
        PyArrayObject* x_array = convert_to_numpy(py_x,"x");
        conversion_numpy_check_type(x_array,PyArray_DOUBLE,"x");
        #define X1(i) (*((double*)(x_array->data + (i)*Sx[0])))
        #define X2(i,j) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1])))
        #define X3(i,j,k) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2])))
        #define X4(i,j,k,l) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2] + (l)*Sx[3])))
        npy_intp* Nx = x_array->dimensions;
        npy_intp* Sx = x_array->strides;
        int Dx = x_array->nd;
        double* x = (double*) x_array->data;
        x_used = 1;
        /*<function call here>*/     
            const int num_points = Nu1[0];    // len(u1)
            int xi;
            
            //#pragma omp parallel for
            for (xi = 0; xi < num_points; xi++) {
                const construct_A();
                const CvMat A_mat = cvMat(4, 3, CV_64F, const_cast<double *>(A));
                
                const construct_b();
                const CvMat b_vec = cvMat(4, 1, CV_64F, const_cast<double *>(b));
                
                CvMat x_vec = cvMat(3, 1, CV_64F, &x_(0));
                
                /* Solve for x vector */
                cvSolve(&A_mat, &b_vec, &x_vec, cv::DECOMP_SVD);
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(u1_used)
    {
        Py_XDECREF(py_u1);
        #undef U11
        #undef U12
        #undef U13
        #undef U14
    }
    if(P1_used)
    {
        Py_XDECREF(py_P1);
        #undef P11
        #undef P12
        #undef P13
        #undef P14
    }
    if(u2_used)
    {
        Py_XDECREF(py_u2);
        #undef U21
        #undef U22
        #undef U23
        #undef U24
    }
    if(P2_used)
    {
        Py_XDECREF(py_P2);
        #undef P21
        #undef P22
        #undef P23
        #undef P24
    }
    if(x_used)
    {
        Py_XDECREF(py_x);
        #undef X1
        #undef X2
        #undef X3
        #undef X4
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* iterative_LS_triangulation(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occurred = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"u1","P1","u2","P2","tolerance","x","x_status","local_dict", NULL};
    PyObject *py_u1, *py_P1, *py_u2, *py_P2, *py_tolerance, *py_x, *py_x_status;
    int u1_used, P1_used, u2_used, P2_used, tolerance_used, x_used, x_status_used;
    py_u1 = py_P1 = py_u2 = py_P2 = py_tolerance = py_x = py_x_status = NULL;
    u1_used= P1_used= u2_used= P2_used= tolerance_used= x_used= x_status_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOOO|O:iterative_LS_triangulation",const_cast<char**>(kwlist),&py_u1, &py_P1, &py_u2, &py_P2, &py_tolerance, &py_x, &py_x_status, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_u1 = py_u1;
        PyArrayObject* u1_array = convert_to_numpy(py_u1,"u1");
        conversion_numpy_check_type(u1_array,PyArray_DOUBLE,"u1");
        #define U11(i) (*((double*)(u1_array->data + (i)*Su1[0])))
        #define U12(i,j) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1])))
        #define U13(i,j,k) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1] + (k)*Su1[2])))
        #define U14(i,j,k,l) (*((double*)(u1_array->data + (i)*Su1[0] + (j)*Su1[1] + (k)*Su1[2] + (l)*Su1[3])))
        npy_intp* Nu1 = u1_array->dimensions;
        npy_intp* Su1 = u1_array->strides;
        int Du1 = u1_array->nd;
        double* u1 = (double*) u1_array->data;
        u1_used = 1;
        py_P1 = py_P1;
        PyArrayObject* P1_array = convert_to_numpy(py_P1,"P1");
        conversion_numpy_check_type(P1_array,PyArray_DOUBLE,"P1");
        #define P11(i) (*((double*)(P1_array->data + (i)*SP1[0])))
        #define P12(i,j) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1])))
        #define P13(i,j,k) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1] + (k)*SP1[2])))
        #define P14(i,j,k,l) (*((double*)(P1_array->data + (i)*SP1[0] + (j)*SP1[1] + (k)*SP1[2] + (l)*SP1[3])))
        npy_intp* NP1 = P1_array->dimensions;
        npy_intp* SP1 = P1_array->strides;
        int DP1 = P1_array->nd;
        double* P1 = (double*) P1_array->data;
        P1_used = 1;
        py_u2 = py_u2;
        PyArrayObject* u2_array = convert_to_numpy(py_u2,"u2");
        conversion_numpy_check_type(u2_array,PyArray_DOUBLE,"u2");
        #define U21(i) (*((double*)(u2_array->data + (i)*Su2[0])))
        #define U22(i,j) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1])))
        #define U23(i,j,k) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1] + (k)*Su2[2])))
        #define U24(i,j,k,l) (*((double*)(u2_array->data + (i)*Su2[0] + (j)*Su2[1] + (k)*Su2[2] + (l)*Su2[3])))
        npy_intp* Nu2 = u2_array->dimensions;
        npy_intp* Su2 = u2_array->strides;
        int Du2 = u2_array->nd;
        double* u2 = (double*) u2_array->data;
        u2_used = 1;
        py_P2 = py_P2;
        PyArrayObject* P2_array = convert_to_numpy(py_P2,"P2");
        conversion_numpy_check_type(P2_array,PyArray_DOUBLE,"P2");
        #define P21(i) (*((double*)(P2_array->data + (i)*SP2[0])))
        #define P22(i,j) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1])))
        #define P23(i,j,k) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1] + (k)*SP2[2])))
        #define P24(i,j,k,l) (*((double*)(P2_array->data + (i)*SP2[0] + (j)*SP2[1] + (k)*SP2[2] + (l)*SP2[3])))
        npy_intp* NP2 = P2_array->dimensions;
        npy_intp* SP2 = P2_array->strides;
        int DP2 = P2_array->nd;
        double* P2 = (double*) P2_array->data;
        P2_used = 1;
        py_tolerance = py_tolerance;
        double tolerance = convert_to_float(py_tolerance,"tolerance");
        tolerance_used = 1;
        py_x = py_x;
        PyArrayObject* x_array = convert_to_numpy(py_x,"x");
        conversion_numpy_check_type(x_array,PyArray_DOUBLE,"x");
        #define X1(i) (*((double*)(x_array->data + (i)*Sx[0])))
        #define X2(i,j) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1])))
        #define X3(i,j,k) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2])))
        #define X4(i,j,k,l) (*((double*)(x_array->data + (i)*Sx[0] + (j)*Sx[1] + (k)*Sx[2] + (l)*Sx[3])))
        npy_intp* Nx = x_array->dimensions;
        npy_intp* Sx = x_array->strides;
        int Dx = x_array->nd;
        double* x = (double*) x_array->data;
        x_used = 1;
        py_x_status = py_x_status;
        PyArrayObject* x_status_array = convert_to_numpy(py_x_status,"x_status");
        conversion_numpy_check_type(x_status_array,PyArray_INT,"x_status");
        #define X_STATUS1(i) (*((int*)(x_status_array->data + (i)*Sx_status[0])))
        #define X_STATUS2(i,j) (*((int*)(x_status_array->data + (i)*Sx_status[0] + (j)*Sx_status[1])))
        #define X_STATUS3(i,j,k) (*((int*)(x_status_array->data + (i)*Sx_status[0] + (j)*Sx_status[1] + (k)*Sx_status[2])))
        #define X_STATUS4(i,j,k,l) (*((int*)(x_status_array->data + (i)*Sx_status[0] + (j)*Sx_status[1] + (k)*Sx_status[2] + (l)*Sx_status[3])))
        npy_intp* Nx_status = x_status_array->dimensions;
        npy_intp* Sx_status = x_status_array->strides;
        int Dx_status = x_status_array->nd;
        int* x_status = (int*) x_status_array->data;
        x_status_used = 1;
        /*<function call here>*/     
            const int num_points = Nu1[0];    // len(u1)
            int xi, i;
            
            //#pragma omp parallel for
            for (xi = 0; xi < num_points; xi++) {
                construct_A();
                CvMat A_mat = cvMat(4, 3, CV_64F, A);
                CvMat A_01_mat = cvMat(2, 3, CV_64F, &A[3 * 0]);    // A[0:2, :]
                CvMat A_23_mat = cvMat(2, 3, CV_64F, &A[3 * 2]);    // A[2:4, :]
                
                construct_b();
                CvMat b_vec = cvMat(4, 1, CV_64F, b);
                CvMat b_01_vec = cvMat(2, 1, CV_64F, &b[1 * 0]);    // b[0:2, :]
                CvMat b_23_vec = cvMat(2, 1, CV_64F, &b[1 * 2]);    // b[2:4, :]
                
                CvMat x_vec = cvMat(3, 1, CV_64F, &x_(0));
                
                /* Init depths */
                double d1, d1_new, d2, d2_new;
                d1 = d2 = 1.;
                
                /* Hartley suggests 10 iterations at most */
                for (i = 0; i < 10; i++) {
                    /* Solve for x vector */
                    cvSolve(&A_mat, &b_vec, &x_vec, cv::DECOMP_SVD);
                    
                    /* Calculate new depths */
                    d1_new = P1_(2, 0) * x_(0) + P1_(2, 1) * x_(1) + P1_(2, 2) * x_(2) + P1_(2, 3);    // P1_(2, :).dot([x_(:), 1.])
                    d2_new = P2_(2, 0) * x_(0) + P2_(2, 1) * x_(1) + P2_(2, 2) * x_(2) + P2_(2, 3);    // P2_(2, :).dot([x_(:), 1.])
                    
                    /* Convergence criterium */
                    if ( ((fabs(d1_new - d1) <= tolerance) && (fabs(d2_new - d2) <= tolerance)) || 
                            ((d1_new == 0) || (d2_new == 0)) ) {
                        x_status[xi] = ((d1_new > 0) && (d2_new > 0));    // points should be in front of both cameras
                        if (d1_new <= 0)
                            x_status[xi] -= 1;    // behind 1st cam
                        if (d2_new <= 0)
                            x_status[xi] -= 2;    // behind 2nd cam
                        break;
                    }
                    
                    /* Re-weight A matrix and b vector with the new depths */
                    cvScale(&A_01_mat, &A_01_mat, 1. / d1_new);    // A[0:2, :] /= d1_new
                    cvScale(&A_23_mat, &A_23_mat, 1. / d2_new);    // A[2:4, :] /= d2_new
                    cvScale(&b_01_vec, &b_01_vec, 1. / d1_new);    // b[0:2, :] /= d1_new
                    cvScale(&b_23_vec, &b_23_vec, 1. / d2_new);    // b[2:4, :] /= d2_new
                    
                    /* Update depths */
                    d1 = d1_new;
                    d2 = d2_new;
                }
            }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occurred = 1;       
    }                                
    /*cleanup code*/                     
    if(u1_used)
    {
        Py_XDECREF(py_u1);
        #undef U11
        #undef U12
        #undef U13
        #undef U14
    }
    if(P1_used)
    {
        Py_XDECREF(py_P1);
        #undef P11
        #undef P12
        #undef P13
        #undef P14
    }
    if(u2_used)
    {
        Py_XDECREF(py_u2);
        #undef U21
        #undef U22
        #undef U23
        #undef U24
    }
    if(P2_used)
    {
        Py_XDECREF(py_P2);
        #undef P21
        #undef P22
        #undef P23
        #undef P24
    }
    if(x_used)
    {
        Py_XDECREF(py_x);
        #undef X1
        #undef X2
        #undef X3
        #undef X4
    }
    if(x_status_used)
    {
        Py_XDECREF(py_x_status);
        #undef X_STATUS1
        #undef X_STATUS2
        #undef X_STATUS3
        #undef X_STATUS4
    }
    if(!(PyObject*)return_val && !exception_occurred)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                


static PyMethodDef compiled_methods[] = 
{
    {"linear_LS_triangulation",(PyCFunction)linear_LS_triangulation , METH_VARARGS|METH_KEYWORDS},
    {"iterative_LS_triangulation",(PyCFunction)iterative_LS_triangulation , METH_VARARGS|METH_KEYWORDS},
    {NULL,      NULL}        /* Sentinel */
};

PyMODINIT_FUNC inittriangulation_ext(void)
{
    
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
    (void) Py_InitModule("triangulation_ext", compiled_methods_with_docs); 
    //    (void) Py_InitModule("triangulation_ext", compiled_methods);
}

#ifdef __CPLUSCPLUS__
}
#endif
