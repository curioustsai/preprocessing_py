f = open("angleRetain.pcm", 'rb')

try:
    byte = f.read(2)
    while byte != "":
        angle = int.from_bytes(byte, "big") * 180 / 32767
        # angle = angle % 360
        print(angle)
        byte = f.read(2)
finally:
    f.close()
