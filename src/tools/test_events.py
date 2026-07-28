from src.services.events import EventsService


def main():
    events = EventsService()

    print("A obter eventos...")
    data = events.list(3)

    print(data)


if __name__ == "__main__":
    main()
