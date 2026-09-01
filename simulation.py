'''HAZARD_CONFIG = {"pedestrian": {"confidence": 0.95, "sensor": "camera" },"vehicle": {"confidence": 0.98,"sensor": "camera"},
                     "obstacle": {"confidence": 0.90,"sensor": "lidar"}
def get_user_event():
    object_type = input("Enter hazard type (pedestrian/vehicle/obstacle): ").lower()
    distance = float(input("Enter distance (meters): "))
    position = input("Enter position (front/left/right): ").lower()
    sensor_status = input(
        "Enter sensor status (active/degraded/failed): ")
    config = HAZARD_CONFIG.get(hazard_type,
    {
        "confidence": 0.70,
        "sensor": "unknown"
    }
    confidence = config["confidence"]
    if sensor_status == "degraded":
        confidence *= 0.6

    elif sensor_status == "failed":
        confidence = 0.0

    event = {
        "type": hazard_type,
        "distance": distance,
        "position": position,
        "confidence": round(confidence, 2),
        "sensor": config["sensor"],
        "sensor_status": sensor_status
    }

    return event
)'''

    
scenario = [
    {
        "time": 0,
        "type": "clear",
        "distance": None,
        "position": "none",
        "confidence": 1.0,
        "sensor_status": "active"
    },
    {
        "time": 5,
        "type": "vehicle",
        "distance": 18,
        "position": "front",
        "confidence": 0.95,
        "sensor_status": "active"
    },
    {
        "time": 10,
        "type": "pedestrian",
        "distance": 6,
        "position": "front",
        "confidence": 0.98,
        "sensor_status": "active"
    },
    {
        "time": 15,
        "type": "obstacle",
        "distance": 5,
        "position": "front",
        "confidence": 0.90,
        "sensor_status": "active"
    },
    {
        "time": 20,
         "type": "sensor_gap",
        "distance": None,
        "position": "unknown",
        "confidence": 0.0,
        "sensor_status": "failed"
    },
    {
        "time": 25,
        "type": "clear",
        "distance": None,
        "position": "none",
        "confidence": 1.0,
        "sensor_status": "active"
    }
]