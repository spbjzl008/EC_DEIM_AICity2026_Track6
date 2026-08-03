"""Class definitions shared by data preparation, training, and inference."""

TRACK6_CLASSES = [
    "Vehicle.Car",
    "Vehicle.Pickup Truck",
    "Vehicle.Single Truck",
    "Vehicle.Combo Truck",
    "Vehicle.Heavy Duty Vehicle",
    "Vehicle.Trailer",
    "Vehicle.Motorcycle",
    "Vehicle.Bicycle",
    "Vehicle.Van",
    "Person",
]

PRETRAIN_CLASSES = TRACK6_CLASSES + ["Vehicle.Truck_Generic", "_Other"]

CLASS_TO_ID = {name: index for index, name in enumerate(PRETRAIN_CLASSES)}
GENERIC_TRUCK_ID = CLASS_TO_ID["Vehicle.Truck_Generic"]
OTHER_ID = CLASS_TO_ID["_Other"]
BICYCLE_ID = CLASS_TO_ID["Vehicle.Bicycle"]
TRAILER_ID = CLASS_TO_ID["Vehicle.Trailer"]
FINE_TRUCK_IDS = frozenset(
    CLASS_TO_ID[name]
    for name in (
        "Vehicle.Pickup Truck",
        "Vehicle.Single Truck",
        "Vehicle.Combo Truck",
        "Vehicle.Heavy Duty Vehicle",
        "Vehicle.Trailer",
    )
)

POSITIVE_CLASS_WEIGHTS = {
    CLASS_TO_ID["Vehicle.Trailer"]: 1.20,
    CLASS_TO_ID["Vehicle.Motorcycle"]: 1.35,
    CLASS_TO_ID["Vehicle.Bicycle"]: 1.35,
    CLASS_TO_ID["Vehicle.Van"]: 1.15,
    CLASS_TO_ID["Person"]: 1.10,
}
