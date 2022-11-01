import xlrd
from xlwt import *


# 多次实验时写入数据代码
def write_data(data_list, sheet_name, iteration):
    for each_i in range(len(data_list)):
        sheet_name.write(each_i, iteration, str(data_list[each_i]))


def create_csv(sheet1, data_list, iteration=1):
    write_data(data_list=data_list, sheet_name=sheet1, iteration=iteration)


def write_to_excel(cost_list, is_one, filename):
    wb = Workbook()
    sheet1 = wb.add_sheet(sheetname='cost')
    if is_one:
        create_csv(sheet1=sheet1, data_list=cost_list)
    else:
        for index in range(len(cost_list)):
            create_csv(sheet1=sheet1, data_list=cost_list[index], iteration=index)
    wb.save(filename)


def read_file(h_file, task_file):
    channel_uc = xlrd.open_workbook(h_file)
    task_file = xlrd.open_workbook(task_file)
    uc_sheet = channel_uc.sheets()[0]
    task_sheet = task_file.sheets()[0]

    return uc_sheet, task_sheet



