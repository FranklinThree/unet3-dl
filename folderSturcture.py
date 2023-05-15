import os

def list_files(start_directory, level=0):
    for entry in os.scandir(start_directory):
        if entry.is_file():
            print(f"{'|   ' * (level - 1)}|--- {entry.name}")
        elif entry.is_dir():
            print(f"{'|   ' * (level - 1)}|--- {entry.name}")
            list_files(entry.path, level + 1)


if __name__ == '__main__':

    start_directory = input("请输入要列出的目录: ")
    print(f'目录 {start_directory} 的树状图结构: ')
    list_files(start_directory)