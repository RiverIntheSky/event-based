#include "Optimizer.h"

namespace ev {

void Optimizer::optimize(shared_ptr<MapPoint> pMP) {
    // for one map point and n keyframes, parameter numbers 3 + n * 5
    int nKFs = pMP->observations();
    int nParams = 3 + nKFs * 5;

    const gsl_multimin_fminimizer_type *T =
      gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer *s = NULL;
    gsl_vector *step_size, *x;
    gsl_multimin_function minex_func;

    size_t iter = 0;
    int status;
    double size;

    /* Starting point */
    x = gsl_vector_alloc(nParams);

    for (int i = 0; i < 3; i ++)
        gsl_vector_set(x, i, w[i]);

    for (int i = 0; i < 3; i ++)
        gsl_vector_set(x, i+3, v[i]);

    for (int i = 0; i < 2; i ++)
        gsl_vector_set(x, i+6, p[i]);

    /* Set initial step sizes to 1 */
    step_size = gsl_vector_alloc(nParams);
    gsl_vector_set_all(step_size, 1);

    /* Initialize method and iterate */
    minex_func.n = nParams;
    minex_func.f = variance;
    minex_func.params = &param;

    s = gsl_multimin_fminimizer_alloc(T, nParams);
    gsl_multimin_fminimizer_set(s, &minex_func, x, step_size);

    do
    {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);

        if (status)
            break;

        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, 1e-2);

        if (status == GSL_SUCCESS)
        {
            printf ("converged to minimum at\n");
        }

        LOG(INFO) << size;

    }
    while (status == GSL_CONTINUE && iter < 100);

    for (int i = 0; i < 3; i++) {
        w[i] = gsl_vector_get(s->x, i);
        v[i] = gsl_vector_get(s->x, i+3);
    }
    for (int i = 0; i < 2; i++) {
        p[i] = gsl_vector_get(s->x, i+6);
    }

    gsl_vector_free(x);
    gsl_vector_free(step_size);
    gsl_multimin_fminimizer_free (s);
}

}
