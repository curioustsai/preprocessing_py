# Just for verification, move to Speech Enhancement Framework
def dc_removal(input_file, output_file):
    [fs, input] = wavfile.read(input_file)
    datalen, nchannel = input.shape
    output = np.zeros(input.shape, dtype=float)
    state_y = np.zeros(nchannel, dtype=float)
    state_x = np.zeros(nchannel, dtype=float)

    a = 0.95
    # H(z) = 1-z^-1/(1-az^-1)
    for i in range(0, datalen):
        output[i, :] = a * state_y + input[i, :] - state_x
        state_x = input[i, :]
        state_y = output[i, :]
    wavfile.write(output_file, fs, output.astype(np.int16))
