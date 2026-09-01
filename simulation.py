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
        "type": "pedestrian",
        "distance": 6,
        "position": "front",
        "confidence": 0.95,
        "sensor": "camera",
        "sensor_status": "active"
    },
    {
        "type": "vehicle",
        "distance": 15,
        "position": "front",
        "confidence": 0.98,
        "sensor": "camera",
        "sensor_status": "active"
    },
    {
        "type": "obstacle",
        "distance": 5,
        "position": "front",
        "confidence": 0.90,
        "sensor": "lidar",
        "sensor_status": "active"
    }
]

def get_event(index):
    return scenario[index]

if __name__ == "__main__":
    for i in range(len(scenario)):
        print(f"Event {i + 1}:")
        print(get_event(i))
        print()

    
