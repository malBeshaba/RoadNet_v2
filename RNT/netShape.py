import osgeo.osr as osr
import shapefile


def initShp(url, encode='utf_8_sig'):
    data_address = url  # 新建数据存放位置
    file = shapefile.Writer(data_address, encoding=encode)
    return file


def setShp(file, points, record):
    # print(S)
    file.line([points])
    file.record(*record)


def over(file, url):
    file.close()
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)  # 4326-GCS_WGS_1984; 4490- GCS_China_Geodetic_Coordinate_System_2000
    wkt = proj.ExportToWkt()
    f = open(url.replace(".shp", ".prj"), 'w')
    f.write(wkt)
    f.close()
    print(url, 'over')


def transfer(net, shf):
    file = initShp(shf, 'utf-8')
    file.fields = net.get_fields()
    for shape, record in net.get_shapes():
        if len(shape) > 0:
            setShp(file, shape, record)
    over(file, shf)
