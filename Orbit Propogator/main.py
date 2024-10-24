import numpy as np
from datetime import datetime
from marshmallow import Schema, fields, ValidationError

from src.orbit import Orbit
from src.bodies import Earth
from src.dataclasses import ClassicalOrbitalElements

class InputSchema(Schema):
    from_state = fields.Bool()
    time = fields.Int(required=True)  # Time in milliseconds
    elements = fields.List(fields.Float(), required=True, validate=lambda x: len(x) == 6)
    propagation_span = fields.Int(required=True)
    propagation_step = fields.Int(required=True)


def propagate_orbit(body):
    """
    Core function to handle propagation logic for both Lambda and Flask handlers.
    """
    # Validate and deserialize input
    schema = InputSchema()
    try:
        body = schema.load(body)
    except ValidationError as err:
        return {"error": err.messages}, 400

    # Set up epoch
    time_ms = body['time']
    t0 = datetime.fromtimestamp(time_ms / 1000.0)

    # Pull out time variables
    span = body['propagation_span']
    signedDt = np.copysign(body['propagation_step'], span)

    # Set up orbit
    elements = body['elements']
    if body['from_state']:
        orbit = Orbit.from_state(np.array(elements), Earth, t0)
    else:
        coes = elements
        sma = coes[0]
        ecc = coes[1]
        inc = coes[2]
        raan = coes[3]
        aop = coes[4]
        ta = coes[5]
        coesSat = ClassicalOrbitalElements(sma, ecc, inc, raan, aop, ta)
        orbit = Orbit.from_coes(coesSat, Earth, t0)

    # Output:
    statesSat = []
    statesGeocSat = []

    # Propagate
    for i in range(abs(span)):
        states, statesGeoc = orbit.propagate(signedDt, 1)
        statesSat.append(states[0].tolist())
        statesGeocSat.append(statesGeoc[0].tolist())

    # Return response data
    return {
        "message": "Ok",
        "statesSat": statesSat,
        "statesGeocSat": statesGeocSat
    }, 200



if __name__ == "__main__":
    # Epoch (Vernal Equinox 2024)
    year = 2024
    month = 3
    day = 20
    hour = 6 
    minute = 0
    second = 0
    t0 = datetime(year, month, day, hour, minute, second)

    # Use the propagate_function above
    body = {
        "from_state": False,
        "time": t0.timestamp() * 1000,
        "elements": [7641.80, 0.00000001, 100.73, 0, 0, 90],
        "propagation_span": 3600,
        "propagation_step": 1
    }

    response, status_code = propagate_orbit(body)

    # Plot the satellite trajectory
    from src.plot import plot_eci
    import matplotlib.pyplot as plt

    statesSat = response["statesSat"]
    statesSat = np.array(statesSat)

    print(f"States: {statesSat.shape}")
    plot_eci(
        [statesSat],
        {
            "cb_axes_color": "k",
            "opacity": 0.5,
            "figsize": (20, 10),
            "title": "Satellite Orbit"
        },
    )
    plt.show()