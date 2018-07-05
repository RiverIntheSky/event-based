#pragma once

#include "MapPoint.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

namespace ev {
class Optimizer
{
public:
    double static variance(const gsl_vector *vec, void *params);
    void static optimize(shared_ptr<MapPoint> pMP);
};
}
