def shenzhennanshan():
    file_name = "2023shenzhennanshan.txt"
    with open(file_name, "w") as f:
        for a in {'A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7'}:
            for i in range(1, 10000):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)


def shenzhen():
    file_name = "2023shenzhen.txt"
    with open(file_name, "w") as f:
        for a in {'A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2'}:
            for i in range(1, 10000):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'C'}:
            for i in range(1, 1000):
                for j in range(1, 10):
                    line = a + "%03d" % i + '-' + str(j) + '\n'
                    f.write(line)
                    print(line)
        for i in range(1, 100):
            line = "%02d\n" % i
            f.write(line)
            print(line)


def guangzhou():
    file_name = "2023guangzhou.txt"
    with open(file_name, "w") as f:
        for a in {'A', 'B', 'C', 'D', 'E', 'F'}:
            for i in range(1, 10000):
                line = a + "%04d\n" % (i)
                f.write(line)
                print(line)
        for i in range(1, 1000):
            line = "%03d\n" % (i)
            f.write(line)
            print(line)


def taiyuan():
    file_name = "2023taiyuan.txt"
    with open(file_name, "w") as f:
        for a in {'A'}:
            for i in range(100, 900):
                line = a + "%03d\n" % i
                f.write(line)
                print(line)
        for a in {'B'}:
            for i in range(1000, 2500):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'C'}:
            for i in range(3000, 4400):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'D'}:
            for i in range(4500, 5300):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'E'}:
            for i in range(5500, 5900):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'F'}:
            for i in range(6000, 7800):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        # -------------------- ------------------------
        for a in {'A'}:
            for i in range(8000, 8200):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'B'}:
            for i in range(8200, 8400):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'C'}:
            for i in range(8400, 8700):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'D'}:
            for i in range(8800, 9000):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'E'}:
            for i in range(9200, 9400):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        for a in {'F'}:
            for i in range(9400, 9900):
                line = a + "%04d\n" % i
                f.write(line)
                print(line)
        # -------------------- ------------------------
        for a in {'A'}:
            for i in range(10000, 10500):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'B'}:
            for i in range(10500, 11200):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'C'}:
            for i in range(11200, 13000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'D'}:
            for i in range(13000, 15000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'E'}:
            for i in range(16000, 18000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'F'}:
            for i in range(19000, 20000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'G'}:
            for i in range(20000, 23000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'L'}:
            for i in range(23000, 26000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'J'}:
            for i in range(26000, 28000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'K'}:
            for i in range(28000, 30000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        # -------------------- ------------------------
        for a in {'A'}:
            for i in range(30000, 31000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'B'}:
            for i in range(31000, 32000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'C'}:
            for i in range(32000, 33000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'D'}:
            for i in range(33000, 34000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'E'}:
            for i in range(34000, 35000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'F'}:
            for i in range(35000, 36000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'G'}:
            for i in range(36000, 37000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'L'}:
            for i in range(37000, 38000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'J'}:
            for i in range(38000, 39000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        for a in {'K'}:
            for i in range(39000, 40000):
                line = a + "%05d\n" % i
                f.write(line)
                print(line)
        # # -------------------- ------------------------
        # for i in range(60000, 70000):
        #     line = "%05d\n" % i
        #     f.write(line)
        #     print(line)


if __name__ == '__main__':
    # shenzhen()
    # guangzhou()
    # shenzhennanshan()
    taiyuan()
