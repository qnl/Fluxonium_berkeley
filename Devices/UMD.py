def devices():
    list_of_devices = [
        {'name': 'Fluxonium #11',
         'E_J': 3.37,
         'E_L': 1.283,
         'E_C': 0.845,
         'junc_area': 130 * 160,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #12',
         'E_J': 1.6,
         'E_L': 0.5,
         'E_C': 0.86,
         'junc_area': 130 * 160,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #13',
         'E_J': 3,
         'E_L': 1,
         'E_C': 0.84,
         'junc_area': 130 * 160,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #14',
         'E_J': 6.4,
         'E_L': 1.6,
         'E_C': 0.83,
         'junc_area': 160 * 180,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #15',
         'E_J': 6.2,
         'E_L': 0.86,
         'E_C': 0.82,
         'junc_area': 190 * 190,
         'array_junc_num': 200,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #16',
         'E_J': 6.2,
         'E_L': 0.86,
         'E_C': 0.82,
         'junc_area': 190 * 190,
         'array_junc_num': 200,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #17',
         'E_J': 5.9,
         'E_L': 0.97,
         'E_C': 0.91,
         'junc_area': 160 * 160,
         'array_junc_num': 200,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #19',
         'E_J': 6.1,
         'E_L': 0.856,
         'E_C': 0.77,
         'junc_area': 150 * 150,
         'array_junc_num': 244,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #20',
         'E_J': 3.49,
         'E_L': 0.886,
         'E_C': 0.915,
         'junc_area': 130 * 140,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #21',
         'E_J': 3.6,
         'E_L': 0.42,
         'E_C': 0.81,
         'junc_area': 0.021e6,
         'array_junc_num': 348,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #22',
         'E_J': 3.4,
         'E_L': 0.41,
         'E_C': 0.83,
         'junc_area': 190 * 190,
         'array_junc_num': 348,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #23',
         'E_J': 2.2,
         'E_L': 0.52,
         'E_C': 0.83,
         'junc_area': 190 * 190,
         'array_junc_num': 200,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #25',
         'E_J': 2.56,
         'E_L': 0.71,
         'E_C': 0.81,
         'junc_area': 130 * 140,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #26',
         'E_J': 2.8,
         'E_L': 0.64,
         'E_C': 0.84,
         'junc_area': 130 * 140,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #27',
         'E_J': 3.9,
         'E_L': 1.27,
         'E_C': 0.85,
         'junc_area': 150 * 150,
         'array_junc_num': 136,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #28',
         'E_J': 4.86,
         'E_L': 1.14,
         'E_C': 0.84,
         'junc_area': 160 * 170,
         'array_junc_num': 136,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #22_2',
         'E_J': 2.82,
         'E_L': 0.35,
         'E_C': 0.85,
         'junc_area': 130 * 140,
         'array_junc_num': 348,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #23_2',
         'E_J': 1.65,
         'E_L': 0.4,
         'E_C': 0.84,
         'junc_area': 130 * 140,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #29',
         'E_J': 3.65,
         'E_L': 0.62,
         'E_C': 0.84,
         'junc_area': 150 * 150,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #30',
         'E_J': 3.96,
         'E_L': 0.98,
         'E_C': 0.85,
         'junc_area': 130 * 140,
         'array_junc_num': 196,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Fluxonium #31',
         'E_J': 3.29,
         'E_L': 0.43,
         'E_C': 0.79,
         'junc_area': 130 * 140,
         'array_junc_num': 400,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Julius 1',
         'E_J': 1.65,
         'E_L': 0.19,
         'E_C': 1.14,
         'junc_area': 130 * 180,
         'array_junc_num': 400,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Julius 2',
         'E_J': 4.45,
         'E_L': 0.79,
         'E_C': 1,
         'junc_area': 130 * 380,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus I_1',
         'E_J': 4.79,
         'E_L': 0.96,
         'E_C': 1.04,
         'junc_area': 320 * 140,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus I_2',
         'E_J': 4.84,
         'E_L': 0.95,
         'E_C': 1.04,
         'junc_area': 320 * 140,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Julius III',
         'E_J': 2.24,
         'E_L': 0.39,
         'E_C': 1.18,
         'junc_area': 180 * 140,
         'array_junc_num': 200,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus III_1',
         'E_J': 3.89,
         'E_L': 0.95,
         'E_C': 0.91,
         'junc_area': 320 * 140,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus III_2',
         'E_J': 4.03,
         'E_L': 0.87,
         'E_C': 1.03,
         'junc_area': 320 * 140,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus V_1',
         'E_J': 4.03,
         'E_L': 0.61,
         'E_C': 1.03,
         'junc_area': 320 * 140,
         'array_junc_num': 128,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus V_2',
         'E_J': 3.89,
         'E_L': 0.71,
         'E_C': 1.03,
         'junc_area': 320 * 140,
         'array_junc_num': 100,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VI_1',
         'E_J': 4.98,
         'E_L': 1.15,
         'E_C': 1,
         'junc_area': 310 * 140,
         'array_junc_num': 108,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VI_2',
         'E_J': 5.32,
         'E_L': 1.61,
         'E_C': 1.03,
         'junc_area': 320 * 140,
         'array_junc_num': 76,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VII_1',
         'E_J': 2.94,
         'E_L': 0.88,
         'E_C': 1.07,
         'junc_area': 200 * 140,
         'array_junc_num': 122,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VII_2',
         'E_J': 3.19,
         'E_L': 0.65,
         'E_C': 1.02,
         'junc_area': 210 * 140,
         'array_junc_num': 178,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VIII_1',
         'E_J': 2.96,
         'E_L': 0.66,
         'E_C': 1.05,
         'junc_area': 210 * 140,
         'array_junc_num': 176,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus VIII_2',
         'E_J': 2.49,
         'E_L': 0.88,
         'E_C': 0.9,
         'junc_area': 200 * 140,
         'array_junc_num': 120,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Julius IV',
         'E_J': 3.55,
         'E_L': 0.9,
         'E_C': 1.03,
         'junc_area': 320 * 140,
         'array_junc_num': 144,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XII_1',
         'E_J': 5.98,
         'E_L': 1.17,
         'E_C': 0.99,
         'junc_area': 155*140,
         'array_junc_num': 144,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XII_2',
         'E_J': 5.39,
         'E_L': 0.67,
         'E_C': 1.26,
         'junc_area': 160*140,
         'array_junc_num': 180,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XIII_1',
         'E_J': 6.2,
         'E_L': 0.77,
         'E_C': 1.0,
         'junc_area': 220 * 140,
         'array_junc_num': 180,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XIII_2',
         'E_J': 6.73,
         'E_L': 1.24,
         'E_C': 1.0,
         'junc_area': 240 * 140,
         'array_junc_num': 112,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XVI_1',
         'E_J': 4.72,
         'E_L': 1.21,
         'E_C': 1.0,
         'junc_area': 130 * 140,
         'array_junc_num': 180,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XVI_2',
         'E_J': 5.2,
         'E_L': 1.98,
         'E_C': 1.0,
         'junc_area': 130 * 140,
         'array_junc_num': 112,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XVIII_1',
         'E_J': 5.84,
         'E_L': 0.45,
         'E_C': 1.0,
         'junc_area': 130 * 110,
         'array_junc_num': 310,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XVIII_2',
         'E_J': 5.88,
         'E_L': 0.72,
         'E_C': 1.0,
         'junc_area': 130 * 110,
         'array_junc_num': 206,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Sirius I',
         'E_J': 9.1,
         'E_L': 0.77,
         'E_C': 1.0,
         'junc_area': 150 * 140,
         'array_junc_num': 206,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Sirius II',
         'E_J': 7.05,
         'E_L': 0.75,
         'E_C': 1.0,
         'junc_area': 130 * 140,
         'array_junc_num': 166,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Sirius III',
         'E_J': 5.58,
         'E_L': 0.65,
         'E_C': 1.0,
         'junc_area': 130 * 140,
         'array_junc_num': 166,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XX_1',
         'E_J': 10.2,
         'E_L': 0.95,
         'E_C': 1.0,
         'junc_area': 140 * 140,
         'array_junc_num': 180,
         'array_junc_area': 2 * 0.4
         },

        {'name': 'Augustus XX_2',
         'E_J': 8.5,
         'E_L': 1.52,
         'E_C': 1.0,
         'junc_area': 140 * 140,
         'array_junc_num': 112,
         'array_junc_area': 2 * 0.4
         }
    ]
    return list_of_devices




