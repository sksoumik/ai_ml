from utils import read_csv_file


def main():
    df = read_csv_file("../../dataset/stack-overflow-data.csv")
    print(df.head())


if __name__ == "__main__":
    main()
