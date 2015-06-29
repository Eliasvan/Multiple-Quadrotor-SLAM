# Tested with autowrap 0.6.1 and Cython 0.21.1 and GTSAM 3.2.1
#
# To build:
#   Run "setup.py",
#   or run:
#       rm -f gtsam.so && autowrap --out gtsam.pyx gtsam_defs.pxd && python setup.py build_ext --inplace
#
# To test:
#   Run the "test.py" file in this directory.

from libcpp cimport bool
from libcpp.string  cimport string as libcpp_string
from smart_ptr cimport shared_ptr



cdef extern from "<gtsam/3rdparty/gtsam_eigen_includes.h>" namespace "Eigen":
    cdef cppclass MatrixXd:
        MatrixXd() except +
        MatrixXd(int rows, int cols) except +
        MatrixXd(MatrixXd other) except +
        int rows()
        int cols()
        double coeff(int row, int col)
    
    cdef cppclass VectorXd:
        VectorXd() except +
        VectorXd(int rows) except +
        VectorXd(VectorXd other) except +
        int rows()
        double coeff(int row)

cdef extern from "<gtsam/base/Matrix.h>" namespace "gtsam":
    ctypedef MatrixXd Matrix

cdef extern from "<gtsam/base/Vector.h>" namespace "gtsam":
    ctypedef VectorXd Vector


cdef extern from "<gtsam/base/types.h>" namespace "gtsam":
    ctypedef size_t Key

cdef extern from "<gtsam/inference/Symbol.h>" namespace "gtsam":
    cdef Key symbol(unsigned char c, size_t j)
    unsigned char symbolChr(Key key)
    size_t symbolIndex(Key key)


cdef extern from "<gtsam/base/Value.h>" namespace "gtsam":
    cdef cppclass Value:
        void print_ "print"()

cdef extern from "<gtsam/base/DerivedValue.h>" namespace "gtsam":
    cdef cppclass DerivedValue[DERIVED](Value):
        # wrap-inherits:
        #   Value
        pass

cdef extern from "<gtsam/nonlinear/Values.h>" namespace "gtsam::Values":
    cdef cppclass Filtered[ValueType]:
        # wrap-instances:
        #   Filtered_Point3 := Filtered[Point3]
        Filtered(Filtered[ValueType]&)

cdef extern from "<gtsam/nonlinear/Values.h>" namespace "gtsam":
    cdef cppclass Values:
        Values() except +
        Values(Values& other) except +
        Values(Filtered[Point3]& other) except +
        void print_ "print"()
        #Value& at(Key j)    # autowrap doesn't allow to return reference variables
        #Pose2& Pose2_at "at"(Key j)    # cast from Value to Pose2
        bool exists(Key j)
        void insert(Key j, Value& val) except +
        void insert(Key j, Point2& val) except +    # implicit cast from Point2 to Value, ...
        void insert(Key j, Point3& val) except +    # ...
        void insert(Key j, Pose2& val) except +     # ...
        void insert(Key j, Pose3& val) except +     # ...
        void insert(Values& values) except +
        Filtered[Point3] filter_Point3 "gtsam::Values::filter<gtsam::Point3>"()
        void clear()


cdef extern from "<gtsam/geometry/Point2.h>" namespace "gtsam":
    cdef cppclass Point2(DerivedValue[Point2]):
        # wrap-inherits:
        #   DerivedValue[Point2]
        Point2() except +
        Point2(double x, double y) except +
        Point2(Vector& v) except +
        double x()
        double y()
        Vector vector()

cdef extern from "<gtsam/geometry/Point3.h>" namespace "gtsam":
    cdef cppclass Point3(DerivedValue[Point3]):
        # wrap-inherits:
        #   DerivedValue[Point3]
        Point3() except +
        Point3(double x, double y, double z) except +
        Point3(Vector& v) except +
        double x()
        double y()
        double z()
        Vector vector()


cdef extern from "<gtsam/geometry/Rot2.h>" namespace "gtsam":
    cdef cppclass Rot2(DerivedValue[Rot2]):
        # wrap-inherits:
        #   DerivedValue[Rot2]
        Rot2() except +
        Rot2(double theta) except +
        double theta()
        Matrix matrix()

cdef extern from "<gtsam/geometry/Rot3.h>" namespace "gtsam":
    cdef cppclass Rot3(DerivedValue[Rot3]):
        # wrap-inherits:
        #   DerivedValue[Rot3]
        Rot3() except +
        Rot3(Rot3& R) except +
        Rot3(double R11, double R12, double R13, double R21, double R22, double R23, double R31, double R32, double R33) except +
        Rot3(Matrix& R) except +
        Matrix matrix()
        Vector quaternion()
    @staticmethod
    cdef Rot3 Rot3_quaternion "gtsam::Rot3::quaternion"(double w, double x, double y, double z) except +
    @staticmethod
    cdef Rot3 Rot3_rodriguez "gtsam::Rot3::rodriguez"(Vector& w, double theta) except +


cdef extern from "<gtsam/geometry/Pose2.h>" namespace "gtsam":
    cdef cppclass Pose2(DerivedValue[Pose2]):
        # wrap-inherits:
        #   DerivedValue[Pose2]
        Pose2() except +
        Pose2(Pose2& pose) except +
        Pose2(double x, double y, double theta) except +
        double x()
        double y()
        double theta()

cdef extern from "<gtsam/geometry/Pose3.h>" namespace "gtsam":
    cdef cppclass Pose3(DerivedValue[Pose3]):
        # wrap-inherits:
        #   DerivedValue[Pose3]
        Pose3() except +
        Pose3(Pose3& pose) except +
        Pose3(Rot3& R, Point3& t) except +
        Pose3(Matrix &T)  except +
        Pose3 inverse()
        Pose3 compose(Pose3& p2)
        Pose3 between(Pose3& p2)
        #Rot3& rotation()
        #Point3& translation()
        Matrix matrix()
        double x()
        double y()
        double z()


cdef extern from "<gtsam/geometry/Cal3_S2.h>" namespace "gtsam":
    cdef cppclass Cal3_S2(DerivedValue[Cal3_S2]):
        # wrap-inherits:
        #   DerivedValue[Cal3_S2]
        Cal3_S2() except +
        Cal3_S2(double fx, double fy, double s, double u0, double v0) except +
        Cal3_S2(double fov, int w, int h) except +
        double fx()
        double fy()
        double skew()
        double px()
        double py()
        Matrix K()

cdef extern from "<gtsam/geometry/Cal3DS2.h>" namespace "gtsam":
    cdef cppclass Cal3DS2(DerivedValue[Cal3DS2]):
        # wrap-inherits:
        #   DerivedValue[Cal3DS2]
        Cal3DS2() except +
        Cal3DS2(double fx, double fy, double s, double u0, double v0, double k1, double k2, double p1, double p2) except +
        double fx()
        double fy()
        double skew()
        double px()
        double py()
        Matrix K()
        double k1()
        double k2()
        double p1()
        double p2()


cdef extern from "<gtsam/geometry/PinholeCamera.h>" namespace "gtsam":
    cdef cppclass PinholeCamera[Calibration]:
        # wrap-instances:
        #   SimpleCamera := PinholeCamera[Cal3_S2]
        #   PinholeCamera_Cal3DS2 := PinholeCamera[Cal3DS2]
        # "wrap-inherits" not used, because autowrap doesn't support multiple annotations for now
        PinholeCamera() except +
        PinholeCamera(Pose3& pose, Calibration& K) except +
        void print_ "print"()
        #Pose3& pose()
        #Calibration& calibration()
        #Point2 project(Point3& pw)
    #ctypedef PinholeCamera[Cal3_S2] SimpleCamera_ "SimpleCamera"
    #@staticmethod
    #cdef SimpleCamera_ SimpleCamera_Level "gtsam::SimpleCamera::Level"(Cal3_S2 &K, Pose2& pose2, double height) except +
    #@staticmethod
    #cdef PinholeCamera[Cal3DS2] PinholeCamera_Cal3DS2_Level "gtsam::PinholeCamera<Cal3DS2>::Level"(Cal3DS2 &K, Pose2& pose2, double height) except +
    #@staticmethod
    #cdef SimpleCamera_ SimpleCamera_Lookat "gtsam::SimpleCamera::Lookat"(Point3& eye, Point3& target, Point3& upVector, Cal3_S2& K) except +
    #@staticmethod
    #cdef PinholeCamera[Cal3DS2] PinholeCamera_Cal3DS2_Lookat "gtsam::PinholeCamera<Calibration>::Lookat"(Point3& eye, Point3& target, Point3& upVector, Cal3DS2& K) except +


cdef extern from "<gtsam/linear/NoiseModel.h>" namespace "gtsam::noiseModel":
    ctypedef shared_ptr[Base] SharedNoiseModel
    cdef cppclass Base:
        void print_ "print"(libcpp_string& s)
    
    ctypedef shared_ptr[Gaussian] SharedGaussian
    cdef cppclass Gaussian(Base):
        # wrap-inherits:
        #   Base
        pass
    
    ctypedef shared_ptr[Diagonal] SharedDiagonal
    cdef cppclass Diagonal(Gaussian):
        # wrap-inherits:
        #   Gaussian
        pass
    @staticmethod
    cdef SharedDiagonal Diagonal_Sigmas "gtsam::noiseModel::Diagonal::Sigmas"(Vector& sigmas)
    
    ctypedef shared_ptr[Constrained] SharedConstrained
    cdef cppclass Constrained(Diagonal):
        # wrap-inherits:
        #   Diagonal
        pass
    
    ctypedef shared_ptr[Isotropic] SharedIsotropic
    cdef cppclass Isotropic(Diagonal):
        # wrap-inherits:
        #   Diagonal
        pass
    @staticmethod
    cdef SharedIsotropic Isotropic_Sigma "gtsam::noiseModel::Isotropic::Sigma"(size_t dim, double sigma)


cdef extern from "<gtsam/nonlinear/NonlinearFactor.h>" namespace "gtsam":
    cdef cppclass NonlinearFactor:
        void print_ "print"()

cdef extern from "<gtsam/nonlinear/NonlinearEquality.h>" namespace "gtsam":
    cdef cppclass NonlinearEquality[VALUE]:
        # wrap-instances:
        #   NonlinearEquality_Point2 := NonlinearEquality[Point2]
        #   NonlinearEquality_Point3 := NonlinearEquality[Point3]
        #   NonlinearEquality_Pose2 := NonlinearEquality[Pose2]
        #   NonlinearEquality_Pose3 := NonlinearEquality[Pose3]
        # "wrap-inherits" not used, because autowrap doesn't support multiple annotations for now
        NonlinearEquality(Key j, VALUE& feasible) except +
        void print_ "print"()

cdef extern from "<gtsam/slam/PriorFactor.h>" namespace "gtsam":
    cdef cppclass PriorFactor[VALUE]:
        # wrap-instances:
        #   PriorFactor_Point2 := PriorFactor[Point2]
        #   PriorFactor_Point3 := PriorFactor[Point3]
        #   PriorFactor_Pose2 := PriorFactor[Pose2]
        #   PriorFactor_Pose3 := PriorFactor[Pose3]
        # "wrap-inherits" not used, because autowrap doesn't support multiple annotations for now
        PriorFactor(Key key, VALUE& prior, SharedGaussian& model) except +    # SharedNoiseModel is not the baseclass of SharedGaussian, ...
        PriorFactor(Key key, VALUE& prior, SharedDiagonal& model) except +
        PriorFactor(Key key, VALUE& prior, SharedConstrained& model) except +
        PriorFactor(Key key, VALUE& prior, SharedIsotropic& model) except +
        void print_ "print"(libcpp_string& s)

cdef extern from "<gtsam/slam/BetweenFactor.h>" namespace "gtsam":
    cdef cppclass BetweenFactor[VALUE]:
        # wrap-instances:
        #   BetweenFactor_Point2 := BetweenFactor[Point2]
        #   BetweenFactor_Point3 := BetweenFactor[Point3]
        #   BetweenFactor_Pose2 := BetweenFactor[Pose2]
        #   BetweenFactor_Pose3 := BetweenFactor[Pose3]
        BetweenFactor(Key key1, Key key2, VALUE& measured, SharedGaussian& model) except +    # SharedNoiseModel is not the baseclass of SharedGaussian, ...
        BetweenFactor(Key key1, Key key2, VALUE& measured, SharedDiagonal& model) except +
        BetweenFactor(Key key1, Key key2, VALUE& measured, SharedConstrained& model) except +
        BetweenFactor(Key key1, Key key2, VALUE& measured, SharedIsotropic& model) except +
        void print_ "print"(libcpp_string& s)


cdef extern from "<gtsam/inference/FactorGraph.h>" namespace "gtsam":
    cdef cppclass FactorGraph[FACTOR]:
        FactorGraph() except +
        void add(shared_ptr[FACTOR]& factor)
        void print_ "print"()
        size_t size()
        bool empty()
        shared_ptr[FACTOR] at(size_t i) except +
        void resize(size_t size)
        void replace(size_t index, shared_ptr[FACTOR] factor)
        size_t nrFactors()

cdef extern from "<gtsam/nonlinear/NonlinearFactorGraph.h>" namespace "gtsam":
    cdef cppclass NonlinearFactorGraph(FactorGraph[NonlinearFactor]):
        # wrap-inherits:
        #   FactorGraph[NonlinearFactor]
        NonlinearFactorGraph() except +
        void add(shared_ptr[PriorFactor[Point2]]& factor)    # NonlinearFactor is not the baseclass of PriorFactor_Point2, ...
        void add(shared_ptr[PriorFactor[Point3]]& factor)    # ...
        void add(shared_ptr[PriorFactor[Pose2]]& factor)     # ...
        void add(shared_ptr[PriorFactor[Pose3]]& factor)     # ...
        double error(Values& c)
        double probPrime(Values& c)


cdef extern from "<gtsam/nonlinear/NonlinearOptimizer.h>" namespace "gtsam":
    cdef cppclass NonlinearOptimizer:
        Values optimize() except +    # should return "const Values&" instead
        double error()
        int iterations()
        Values values()    # should return "const Values&" instead

cdef extern from "<gtsam/nonlinear/LevenbergMarquardtOptimizer.h>" namespace "gtsam":
    cdef enum VerbosityLM "gtsam::LevenbergMarquardtParams::VerbosityLM":
        SILENT = 0
        TERMINATION
        LAMBDA
        TRYLAMBDA
        TRYCONFIG
        DAMPED
        TRYDELTA
    
    cdef cppclass LevenbergMarquardtParams:
        LevenbergMarquardtParams() except +
        void print_ "print"()
        double lambdaInitial
        double lambdaFactor
        double lambdaUpperBound
        double lambdaLowerBound
        libcpp_string getVerbosityLM()
        VerbosityLM verbosityLM
        double minModelFidelity
        libcpp_string logFile
        bool diagonalDamping
        bool reuse_diagonal_
        bool useFixedLambdaFactor_
        double min_diagonal_
        double max_diagonal_
    
    cdef cppclass LevenbergMarquardtOptimizer(NonlinearOptimizer):
        # wrap-inherits:
        #   NonlinearOptimizer
        LevenbergMarquardtOptimizer(NonlinearFactorGraph& graph, Values& initialValues) except +
        LevenbergMarquardtOptimizer(NonlinearFactorGraph& graph, Values& initialValues, LevenbergMarquardtParams& params) except +
        void print_ "print"()


cdef extern from "gtsam/nonlinear/Marginals.h" namespace "gtsam":
    cdef cppclass Marginals:
        Marginals(NonlinearFactorGraph& graph, Values& solution)
        void print_ "print"()
        Matrix marginalCovariance(Key variable)


cdef extern from "<gtsam/nonlinear/NonlinearISAM.h>" namespace "gtsam":
    cdef cppclass NonlinearISAM:
        NonlinearISAM() except +
        NonlinearISAM(int reorderInterval) except +
        Values estimate() except +
        Matrix marginalCovariance(Key key)
        void print_ "print"()
        void printStats()
        void saveGraph(libcpp_string& s) except +
        void update(NonlinearFactorGraph& newFactors, Values& initialValues) except +
        void reorder_relinearize() except +


cdef extern from "<gtsam/nonlinear/ISAM2.h>" namespace "gtsam":
    cdef enum Factorization "gtsam::ISAM2Params::Factorization":
        CHOLESKY
        QR
    
    cdef cppclass ISAM2Params:
        ISAM2Params() except +
        void print_ "print"()
        int relinearizeSkip
        bool enableRelinearization
        bool evaluateNonlinearError
        libcpp_string getFactorization()
        Factorization factorization
        bool cacheLinearizedFactors
        bool enableDetailedResults
        bool enablePartialRelinearizationCheck
        bool findUnusedFactorSlots
    
    cdef struct ISAM2Result:
        void print_ "print"()
        size_t variablesRelinearized
        size_t variablesReeliminated
        size_t factorsRecalculated
        size_t cliques
    
    cdef cppclass ISAM2:
        ISAM2(ISAM2Params& params) except +
        ISAM2() except +
        void update() except +    # autowrap doesn't have a convertor for "struct"s yet, so using "void"
        void update(NonlinearFactorGraph& newFactors, Values& newTheta) except +
        #ISAM2Result update() except +
        #ISAM2Result update(NonlinearFactorGraph& newFactors, Values& newTheta) except +
        Values calculateEstimate() except +
        #Point2 calculateEstimate(Key key) except +
        #Point3 calculateEstimate(Key key) except +
        #Pose2 calculateEstimate(Key key) except +
        #Pose3 calculateEstimate(Key key) except +
        Values calculateBestEstimate() except +
        Matrix marginalCovariance(Key key)
        void print_ "print"()
        void printStats()
        size_t lastAffectedVariableCount
        size_t lastAffectedFactorCount
        size_t lastAffectedCliqueCount
        size_t lastAffectedMarkedCount
        size_t lastBacksubVariableCount
        size_t lastNnzTop
