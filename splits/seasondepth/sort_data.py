if __name__ == "__main__":
    src_file = 'val_files_ori.txt'
    des_file = 'val_files.txt'
    lines = []
    with open(src_file) as f:
        lines = f.readlines()
        lines.sort()

    with open(des_file,'w') as g:
        for line in lines:
            g.write(line)