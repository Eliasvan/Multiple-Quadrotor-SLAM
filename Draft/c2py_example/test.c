/*
 * Warning, this C file is not directly compilable on its own,
 * use the "convert_c_to_ext_lib.py" module to generate the library.
 * It only serves to properly highlight the syntax.
 */



/* Libraries */

/* m */    // "libm" is the math library



/* Includes */

#include <stdint.h>
#include <stdio.h>
#include <math.h>



/* Support code */

int counter = 0;

double calc_tan(double a)
{
    counter++;
    return tan(a);
}



/* Functions exported to Python */

/* Some text about func1.
 * 
 * Arguments:
 *      a = float()    # [in]
 *      b = float()    # [in]
 * 
 *      buffer = np.empty((3, 2), dtype=np.int32)    # [out]
 */
void func1(/* ... */)
{
    double c = calc_tan(a) + b;
    printf ("%f, %f; %f\n", a, b, c);
    
    int32_t ci = c * 1000.0;
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 2; j++)
            // Note that "BUFFER2()" is a macro defined outside this C file, converted from "buffer".
            BUFFER2(i, j) = ci + i*2 + j;
            // buffer[i*2 + j] = ci + i*2 + j;    // only use this if "buffer"'s data is aligned
    printf ("%d, %d\n", Sbuffer[0], Sbuffer[1]);
}

void func2()
{
    double a = M_PI;
    printf ("%f, %f; counter = %d\n", a, cos(a), counter);
    
    return_val = a;
}
