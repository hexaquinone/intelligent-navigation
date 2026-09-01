from simulation import get_all_events
from brain import make_decision
from metrics import update_metrics


def run_simulation():

    events = get_all_events()

    results = []

    for event in events:

        # Brain analyses the sensor event
        decision = make_decision(event)

        # Update metrics
        update_metrics(event, decision)

        # Store combined result
        results.append({
            "event": event,
            "decision": decision
        })

    return results


if __name__ == "__main__":

    results = run_simulation()

    for result in results:

        print("\nEVENT:")
        print(result["event"])

        print("\nDECISION:")
        print(result["decision"])

        from simulation import get_all_events
        from brain import make_decision 
        from metrices import record_event, get_metrics 
        def run_trip(): 
            events = get_all_events() 
            print("=" * 60) 
            print("INTELLIGENT NAVIGATION SYSTEM") 
            print("=" * 60) 
            for event in events: 
                # Get event from simulation 
                print(f"\nTime: {event['time']} sec") 
                print(f"Detection: {event['type']}") 
                print(f"Position: {event['position']}") 
                print(f"Distance: {event['distance']}") 
                print(f"Weather: {event['weather']}") 
                print(f"Visibility: {event['visibility']}%") 
                # Send event to decision engine decision = make_decision(event) 
                # Show decision 
                print(f"Risk: {decision['risk']}") 
                print(f"Action: {decision['action']}") 
                print(f"Reason: {decision['reason']}") 
                # Send event + risk to metrics 
                record_event( event, risk=decision["risk"] ) 
                # Show final metrics 
                print("\n" + "=" * 60) 
                print("TRIP METRICS") 
                print("=" * 60) 
                final_metrics = get_metrics() 
                for key, value in final_metrics.items():
                     print(f"{key}: {value}") 
                if __name__ == "__main__": 
                    run_trip()