import os, re, wget, sys, cv2
import threading, datetime
from lxml import etree
import mysql.connector as mysql
from dateutil.rrule import rrule, DAILY


def write_xml(np_img, objects, dst_path):
    img_height, img_width, img_depth = np_img.shape
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = "img"
    etree.SubElement(annotation, "filename").text = objects['img_name']
    etree.SubElement(annotation, "path").text = "tmp/img"
    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "Unknown"
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(img_width)
    etree.SubElement(size, "height").text = str(img_height)
    etree.SubElement(size, "depth").text = str(img_depth)
    etree.SubElement(annotation, "segmented").text = '0'
    draw = np_img.copy()
    if objects['boxes'] is not None:
        for box in objects['boxes']:
            key_object = etree.SubElement(annotation, "object")
            etree.SubElement(key_object, "name").text = box['type']
            etree.SubElement(key_object, "pose").text ='pose'
            etree.SubElement(key_object, "truncated").text = str(0)
            etree.SubElement(key_object, "difficult").text = str(0)
            bnd_box = etree.SubElement(key_object, "bndbox")
            etree.SubElement(bnd_box, "xmin").text = str(box['xmin']) if box['xmin'] > 0 else str(0)
            etree.SubElement(bnd_box, "ymin").text = str(box['ymin']) if box['ymin'] > 0 else str(0)
            etree.SubElement(bnd_box, "xmax").text = str(box['xmax']) if box['xmax'] < img_width else str(img_width)
            etree.SubElement(bnd_box, "ymax").text = str(box['ymax']) if box['ymax'] < img_height else str(img_height)
            draw = cv2.rectangle(draw, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])),
                                 color=[0, 0, 255], thickness=10)
        m_xml_dir = os.path.split(dst_path)[0]
        draw_dir = m_xml_dir.replace('Annotations', 'Show')
        if not os.path.exists(draw_dir):
            os.makedirs(draw_dir, exist_ok=True)
        cv2.imwrite(os.path.join(draw_dir, objects['img_name']), draw)

    doc = etree.ElementTree(annotation)
    doc.write(dst_path, pretty_print=True)

def get_polygon_info(m_beg_date, beg_time, m_end_date, end_time):
    db = mysql.connect(
        host = "10.88.1.80",
        user = "inspect",
        passwd = "inspect",
        database = "inspect",
        auth_plugin="mysql_native_password",
    )
    print("Connect to Database")
    cursor = db.cursor()
    query = "SELECT capture_url, region, rectify, audit, event_type, time, time FROM inspect2 " \
            "where time>='%s %s' and time<='%s %s' ORDER BY time DESC"%(m_beg_date, beg_time, m_end_date, end_time)

    cursor.execute(query)
    ## fetching all records from the 'cursor' object
    print("Fetch query data")
    records = cursor.fetchall()
    db.close()
    xml_list = list()
    for record in records:
        if record[4] not in [1, 2]:
            continue
        if record[0] is None:
            continue
        # we could delete this for all box pic recored[3] == 0
        # if record[3] not in [-1]:
        #     continue
        poly_str = record[1]
        if poly_str is not None and len(poly_str) > 4:
            poly_list = poly_str.split('"polygon"')[1:]
            boxes_list = list()
            for poly in poly_list:
                point_list = re.findall(r"{(.+?)}", poly)
                point_x_list, point_y_list = list(), list()
                for point in point_list:
                    point_x, point_y = point.split(',')
                    point_x_list.append(float(point_x.split(':')[-1]))
                    point_y_list.append(float(point_y.split(':')[-1]))
                if len(poly.split('type')) > 6:
                    ddb_type = 'zlc'
                else:
                    ddb_type = 'zlc'
                print(len(boxes_list))
                print('[*]'*10)
                boxes_list.append({'xmin': min(point_x_list),
                                   'ymin': min(point_y_list),
                                   'xmax': max(point_x_list),
                                   'ymax': max(point_y_list),
                                   'type': ddb_type})
            img_name = os.path.split(record[0])[1]
            xml_name = os.path.splitext(img_name)[0] + '.xml'
            xml_list.append({
                'img_url': record[0],
                'xml_name': xml_name,
                'img_name': img_name,
                'boxes': boxes_list,
                'audit': record[3]
            })

    return xml_list

def download_thread(thread_name, records, m_img_dir, m_xml_dir):
    print(thread_name)
    for m_idx, record in enumerate(records):
        img_path = os.path.join(m_img_dir, record['img_name'])
        xml_path = os.path.join(m_xml_dir, record['xml_name'])
        try:
            if not os.path.exists(img_path):
                wget.download(url=record['img_url'], out=img_path)
                print("Downloading {} finished".format(record['img_name']))

            if os.path.exists(img_path):
                m_img = cv2.imread(img_path)
                if m_img is None:
                    os.system('rm -rf {}'.format(img_path))
                elif record['boxes'] is not None:
                    write_xml(m_img, record, xml_path)
        except:
            print("{} download failed".format(record['img_name']))
            continue
        print("finished: {} , all need download image: {}".format(m_idx, len(records)))
        sys.stdout.flush()


if __name__ == '__main__':

    beg_date = '2020-09-09'
    end_date = '2020-09-12'
    root_path ='/defaultShare/share/wujl/83/online_data/daydata'

    date_str = beg_date.split('-')
    beg_date = datetime.date(int(date_str[0]), int(date_str[1]), int(date_str[2]))
    date_str = end_date.split('-')
    end_date = datetime.date(int(date_str[0]), int(date_str[1]), int(date_str[2]))

    for dt in rrule(DAILY, dtstart=beg_date, until=end_date):
        date_time = dt.strftime("%Y-%m-%d")
        boxes_info = get_polygon_info(m_beg_date=date_time,
                                      beg_time='00:00:00',
                                      m_end_date=date_time,
                                      end_time='23:59:59')
        if len(boxes_info) == 0:
            continue

        if os.path.exists(os.path.join(root_path, date_time)):
            os.system('rm -rf {}'.format(os.path.join(root_path, date_time)))
        img_dir = os.path.join(root_path, date_time, 'JPEGImages')
        xml_dir = os.path.join(root_path, date_time, 'Annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)

        record_num = len(boxes_info)
        thread_num = 24
        num_per_thread = record_num // thread_num
        if record_num % num_per_thread > 0:
            thread_num += 1
        print("Thread_Num: {}, Record_Num: {}".format(thread_num, record_num))
        thread = []
        for i in range(thread_num):
            if i == thread_num - 1:
                thread_records = boxes_info[i * num_per_thread:]
            else:
                thread_records = boxes_info[i * num_per_thread:(i + 1) * num_per_thread]
            t = threading.Thread(target=download_thread,
                                 args=("thread_{}d".format(i), thread_records, img_dir, xml_dir))
            thread.append(t)
        for t in thread:
            t.start()
        for t in thread:
            t.join()
