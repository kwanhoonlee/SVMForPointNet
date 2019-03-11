import OCC.gp
import OCC.Geom
import OCC.Bnd
import OCC.BRepBndLib
import OCC.BRep
import OCC.BRepPrimAPI
import OCC.BRepAlgoAPI
import OCC.BRepBuilderAPI
import OCC.GProp
import OCC.BRepGProp
import OCC.TopoDS
import OCC.TopExp
import OCC.TopAbs
import ifcopenshell
import ifcopenshell.geom
import operator
import numpy as np
import pandas as pd
import datetime
import os
np.warnings.filterwarnings('ignore')


def get_features(ifctype, path):

    
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    settings.set(settings.USE_BREP_DATA, True)
    settings.set(settings.SEW_SHELLS, True)

    # settings.set(settings.USE_BREP_DATA,True)
    # settings.set(settings.SEW_SHELLS,True)
    # settings.set(settings.USE_WORLD_COORDS,True)

    def get_shape(elem):
        return ifcopenshell.geom.create_shape(settings, elem)

    def get_volume(s):
        props = OCC.GProp.GProp_GProps()
        OCC.BRepGProp.brepgprop_VolumeProperties(s.geometry, props)
        return props.Mass()

    def get_area(s):
        props = OCC.GProp.GProp_GProps()
        OCC.BRepGProp.brepgprop_SurfaceProperties(s.geometry, props)
        return props.Mass()

    def normalize(li):
        mean, std = np.mean(li), np.std(li)
        return map(lambda v: abs(v - mean) / std, li)

    def get_linear(s):
        props = OCC.GProp.GProp_GProps()
        OCC.BRepGProp.brepgprop_LinearProperties(s.geometry, props)
        return props.Mass()

    def get_global_id(elem):
        return elem[0]

    def get_gyration(s):
        props = OCC.GProp.GProp_GProps()
        OCC.BRepGProp.brepgprop_VolumeProperties(s.geometry, props)
        gyradius = props.RadiusOfGyration(OCC.gp.gp_Ax1(props.CentreOfMass(),
                                                        OCC.gp.gp_DZ()))
        return gyradius

    def get_boundingbox(shape, tol=1e-1, as_vec=True):
        bbox = OCC.Bnd.Bnd_Box()
        bbox.SetGap(tol)
        OCC.BRepBndLib.brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        vec = [OCC.gp.gp_Vec(xmin, ymin, zmin), OCC.gp.gp_Vec(xmax, ymax, zmax)]
        X = (vec[0].X() - vec[1].X()) * (-1)
        Y = (vec[0].Y() - vec[1].Y()) * (-1)
        Z = (vec[0].Z() - vec[1].Z()) * (-1)
        return (X, Y, Z)


    def get_type(elem):
        return (elem.is_a())

    def get_ax1(elem):  # area/volume
        return (get_area(elem) / get_volume(elem))

    def get_ax2(elem):  # gyration/volume
        return (get_gyration(elem) / get_volume(elem))

    def get_topods(elem):
        return elem[1]

    def get_X(elem):
        return elem[0]

    def get_Y(elem):
        return elem[1]

    def get_Z(elem):
        return elem[2]

    file = ifcopenshell.open(path)
    ifc = file.by_type(ifctype)
    shapes = list(map(get_shape, ifc))
    volumes = list(map(get_volume, shapes))
    areas = list(map(get_area, shapes))
    # linear = list(map(get_linear, shapes))
    global_ids = list(map(get_global_id, ifc))
    gyrations = list(map(get_gyration, shapes))
    types = list(map(get_type, ifc))
    ax1s = list(map(get_ax1, shapes))
    ax2s = list(map(get_ax2, shapes))
    topo = list(map(get_topods, shapes))
    bounding_value = list(map(get_boundingbox, topo))
    bat_ids = [str(i).split('=')[0] for i in ifc]
    X = list(map(get_X, bounding_value))
    Y = list(map(get_Y, bounding_value))
    Z = list(map(get_Z, bounding_value))

    data = {'areas': areas, 'volumes': volumes, 'gyrations': gyrations,
            'global_ids': global_ids, 'bat_ids':bat_ids, 'types': types, 'ax1s': ax1s, 'ax2s': ax2s,
            'X': X, 'Y': Y, 'Z': Z}

    frame = pd.DataFrame(data)
    return frame

duplex_window = get_features('ifcwindow', "./ifc/SVM/yonsei.ifc")
duplex_window.to_csv("./result/duplex_window_yonsei.csv")

def get_building_ifc(path):
    column = get_features('IfcColumn', path)
    beam = get_features('IfcBeam', path)
    slab = get_features('IfcSlab', path)
    wall = get_features('IfcWallStandardCase', path)
    covering = get_features('IfcCovering', path)
    door = get_features('IfcDoor', path)
    window = get_features('IfcWindow', path)
    railing = get_features('IfcRailing', path)
    df = pd.concat([column, beam, slab, wall, covering, door, window, railing])
    return df


duplex_all = get_building_ifc("./ifc/SVM/yonsei.ifc")
duplex_all.to_csv("./result/duplex_all_yonsei.csv")
