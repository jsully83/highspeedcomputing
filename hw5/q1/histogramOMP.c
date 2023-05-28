#







    // generate N random numbers onthe host
    for (int i = 0; i < N; i++) {
        h_nums[i] = rand() % N_RANGE + 1;
    }

    // set bins to zero
    for (int i = 0; i < N_BINS; i++) {
        h_bins[i] = 0;
    }
