#pragma once

#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>

namespace ev {
    struct Event {
        /// \brief Default constructor.
        Event(): x(), y(), z(){}

        // Event(const Event &event);

        /// \brief Constructor.
        Event(double x_, double y_, double z_, bool p_) {}

    // private:
        double x;
        double y;
        double z;
        bool p;          ///< Polarity, negative or positive
    };

    typedef okvis::Measurement<Event> EventMeasurement;
    typedef std::deque<EventMeasurement, Eigen::aligned_allocator<EventMeasurement> > EventMeasurementDeque;

    struct eventFrameMeasurement {
        std::vector<EventMeasurement> events;
        unsigned counter_w{0}; // counter for temporal window size

        eventFrameMeasurement() {}
    };
}
