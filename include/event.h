#pragma once

#include <okvis/Measurements.h>

namespace ev {
    struct Event {
        /// \brief Default constructor.
        Event(): x(), y(), z(){}

        /// \brief Constructor.
        Event(double x_, double y_, double z_, bool p_) {}

        double x;
        double y;
        double z;        ///< Depth of the event; was designed for depth estimation, not implemented, set to 1
        bool p;          ///< Polarity, negative or positive
    };

    typedef okvis::Measurement<Event> EventMeasurement;
    typedef std::deque<EventMeasurement, Eigen::aligned_allocator<EventMeasurement> > EventMeasurementDeque;

    struct eventFrameMeasurement {
        std::vector<EventMeasurement> events;

        eventFrameMeasurement() {}
    };
}
