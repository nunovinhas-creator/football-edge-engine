from src.api.client import BzzoiroClient


def main():

    client = BzzoiroClient()

    print("Testing Bzzoiro API...")

    data = client.get("events/?limit=1")

    print("Connection successful!")

    print(data)


if __name__ == "__main__":
    main()
