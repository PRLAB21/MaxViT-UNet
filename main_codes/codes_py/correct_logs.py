import json

def correct_logs():
    print('[start]')
    log_file_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet-pvt/setting2/20220410_140450.log.json'
    log_write_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet-pvt/setting2/20220410_140450.log-corrected.json'
    log_lines = open(log_file_path, 'r').readlines()
    log_write_lines = []
    print('len(log_lines) =', len(log_lines))

    for line in log_lines[1:]:
        json_log = json.loads(line)
        if 'iter' in json_log and json_log['iter'] <= 2310:
            log_write_lines.append(line)
        print(json_log)
        # break
    print('len(log_write_lines) =', len(log_write_lines))
    with open(log_write_path, 'w') as log_write_file:
        for log_line in log_write_lines:
            log_write_file.write(log_line)
    print('[end]')
