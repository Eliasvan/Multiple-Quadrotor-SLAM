/*
 * Warning, this C file is not directly compilable on its own,
 * run the "setup.py" file to generate the library.
 * It only serves to properly highlight the syntax.
 */



/* Libraries */

/* opencv_imgproc */



/* Includes */

#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>



/* Support code */

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



/* Functions exported to Python */

/* Arguments:
 *      u1 = np.empty((0, 2), dtype=np.float64)    # [in]
 *      P1 = np.empty((3, 4), dtype=np.float64)    # [in]
 *      u2 = np.empty((0, 2), dtype=np.float64)    # [in]
 *      P2 = np.empty((3, 4), dtype=np.float64)    # [in]
 * 
 *       x = np.empty((0, 3), dtype=np.float64)    # [out]
 * 
 *  Note 1:
 *      The dimensions with size 0 should be replaced by the amount of points to triangulate.
 *  Note 2:
 *      The data of each numpy array argument "a" should be aligned,
 *      to check whether this is the case, the following code shouldn't raise an exception:
 *      #    v = a.view(); v.shape = (a.size,)
 *      To fix this automatically, run:
 *      #    a = a.reshape(a.size).reshape(a.shape)
 */
void linear_LS_triangulation(/* ... */)
{
    const int num_points = Nu1[0];    // len(u1)
    int xi;
    
    #pragma omp parallel for
    for (xi = 0; xi < num_points; xi++) {
        const construct_A();
        const CvMat A_mat = cvMat(4, 3, CV_64F, const_cast<double *>(A));
        
        const construct_b();
        const CvMat b_vec = cvMat(4, 1, CV_64F, const_cast<double *>(b));
        
        CvMat x_vec = cvMat(3, 1, CV_64F, &x_(0));
        
        /* Solve for x vector */
        cvSolve(&A_mat, &b_vec, &x_vec, cv::DECOMP_SVD);
    }
}

/* Arguments:
 *             u1 = np.empty((0, 2), dtype=np.float64)     # [in]
 *             P1 = np.empty((3, 4), dtype=np.float64)     # [in]
 *             u2 = np.empty((0, 2), dtype=np.float64)     # [in]
 *             P2 = np.empty((3, 4), dtype=np.float64)     # [in]
 *      tolerance = float()                                # [in]
 * 
 *             x = np.empty((0, 3), dtype=np.float64)    # [out]
 *      x_status = np.empty( 0    , dtype=np.int32)      # [out]
 * 
 *  Note 1:
 *      The dimensions with size 0 should be replaced by the amount of points to triangulate.
 *  Note 2:
 *      The data of each numpy array argument "a" should be aligned,
 *      to check whether this is the case, the following code shouldn't raise an exception:
 *      #    v = a.view(); v.shape = (a.size,)
 *      To fix this automatically, run:
 *      #    a = a.reshape(a.size).reshape(a.shape)
 */
void iterative_LS_triangulation(/* ... */)
{
    const int num_points = Nu1[0];    // len(u1)
    int xi, i;
    
    #pragma omp parallel for
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
        
        /* Set status */
        x_status[xi] = ( (i < 10) &&                          // points should have converged by now
                         ((d1_new > 0) && (d2_new > 0)) );    // points should be in front of both cameras
        if (d1_new <= 0)
            x_status[xi] -= 1;    // behind 1st cam
        if (d2_new <= 0)
            x_status[xi] -= 2;    // behind 2nd cam
    }
}
