#ifndef UTILS_H
#define UTILS_H

#include "parameters.h"

namespace ev {
class parameterReader
{
public:
    parameterReader(const std::string& filename);
    bool getParameter(ev::Parameters& parameter);

    ev::Parameters parameters;
};
}

#endif // UTILS_H
