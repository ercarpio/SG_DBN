import os

for root, subFolders, files in os.walk('../labels/'):
    for f in files:
        noise_labels = ''
        with open(os.path.join(root, f), 'r') as txt_file:
            for line in txt_file:
                id = ''
                if 'command_s' in line:
                    id = '0'
                elif 'prompt_s' in line:
                    id = '1'
                if id != '':
                    data = line.split(' ')
                    noise_labels += ('noise_' + id + '_s ' + str(float(data[1]) - 0.85) + '\nnoise_' +
                                     id + '_e ' + str(float(data[1]) - 0.85 + 6.25) + '\n')
        with open(os.path.join(root, f), 'a') as txt_file:
            if noise_labels != '':
                txt_file.write(noise_labels)
