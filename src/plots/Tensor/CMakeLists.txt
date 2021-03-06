# DO NOT EDIT THIS FILE! THIS FILE IS AUTOMATICALLY GENERATED BY xml2cmake
PROJECT(Tensor_plot)

INCLUDE(${VISIT_SOURCE_DIR}/CMake/PluginMacros.cmake)

ADD_PLOT_CODE_GEN_TARGETS(Tensor)

SET(COMMON_SOURCES
TensorPluginInfo.C
TensorCommonPluginInfo.C
TensorAttributes.C
)

SET(LIBI_SOURCES
TensorPluginInfo.C
)

SET(LIBG_SOURCES
TensorGUIPluginInfo.C
QvisTensorPlotWindow.C
${COMMON_SOURCES}
)
SET(LIBG_MOC_SOURCES
QvisTensorPlotWindow.h
)

SET(LIBV_SOURCES
TensorViewerEnginePluginInfo.C
TensorViewerPluginInfo.C
avtTensorPlot.C
avtTensorFilter.C
${COMMON_SOURCES}
)

SET(LIBE_SOURCES
TensorViewerEnginePluginInfo.C
TensorEnginePluginInfo.C
avtTensorPlot.C
avtTensorFilter.C
${COMMON_SOURCES}
)


IF(VISIT_PYTHON_SCRIPTING)
    SET(PYINCLUDES ${PYTHON_INCLUDE_PATH} ${VISIT_INCLUDE_DIR}/visitpy/visitpy)
ENDIF(VISIT_PYTHON_SCRIPTING)

INCLUDE_DIRECTORIES(
${CMAKE_CURRENT_SOURCE_DIR}
${VISIT_COMMON_INCLUDES}
${VISIT_INCLUDE_DIR}/avt/DBAtts/MetaData
${VISIT_INCLUDE_DIR}/avt/DBAtts/SIL
${VISIT_INCLUDE_DIR}/avt/Database/Database
${VISIT_INCLUDE_DIR}/avt/FileWriter
${VISIT_INCLUDE_DIR}/avt/Filters
${VISIT_INCLUDE_DIR}/avt/IVP
${VISIT_INCLUDE_DIR}/avt/Math
${VISIT_INCLUDE_DIR}/avt/Pipeline/AbstractFilters
${VISIT_INCLUDE_DIR}/avt/Pipeline/Data
${VISIT_INCLUDE_DIR}/avt/Pipeline/Pipeline
${VISIT_INCLUDE_DIR}/avt/Pipeline/Sinks
${VISIT_INCLUDE_DIR}/avt/Pipeline/Sources
${VISIT_INCLUDE_DIR}/avt/Plotter
${VISIT_INCLUDE_DIR}/avt/QtVisWindow
${VISIT_INCLUDE_DIR}/avt/View
${VISIT_INCLUDE_DIR}/avt/VisWindow/Colleagues
${VISIT_INCLUDE_DIR}/avt/VisWindow/Interactors
${VISIT_INCLUDE_DIR}/avt/VisWindow/Proxies
${VISIT_INCLUDE_DIR}/avt/VisWindow/Tools
${VISIT_INCLUDE_DIR}/avt/VisWindow/VisWindow
${VISIT_INCLUDE_DIR}/gui
${VISIT_INCLUDE_DIR}/viewer/core
${VISIT_INCLUDE_DIR}/viewer/main
${VISIT_INCLUDE_DIR}/viewer/main/ui
${VISIT_INCLUDE_DIR}/viewer/proxy
${VISIT_INCLUDE_DIR}/viewer/rpc
${VISIT_INCLUDE_DIR}/winutil
${VISIT_INCLUDE_DIR}/visit_vtk/full
${VISIT_INCLUDE_DIR}/visit_vtk/lightweight
${VTKh_INCLUDE_DIRS}
${VTKM_DIR}/include/vtkm-1.2
${VTKM_DIR}/include/vtkm-1.2/vtkm/thirdparty/taotuple
${VTK_INCLUDE_DIRS}
${PYINCLUDES}
)

LINK_DIRECTORIES(${VISIT_LIBRARY_DIR} )

ADD_LIBRARY(ITensorPlot ${LIBI_SOURCES})
TARGET_LINK_LIBRARIES(ITensorPlot visitcommon)
SET(INSTALLTARGETS ITensorPlot)

IF(NOT VISIT_SERVER_COMPONENTS_ONLY AND NOT VISIT_ENGINE_ONLY AND NOT VISIT_DBIO_ONLY)
    QT_WRAP_CPP(GTensorPlot LIBG_SOURCES ${LIBG_MOC_SOURCES})
    ADD_LIBRARY(GTensorPlot ${LIBG_SOURCES})
    TARGET_LINK_LIBRARIES(GTensorPlot visitcommon gui )

    ADD_LIBRARY(VTensorPlot ${LIBV_SOURCES})
    TARGET_LINK_LIBRARIES(VTensorPlot visitcommon viewer )

    SET(INSTALLTARGETS ${INSTALLTARGETS} GTensorPlot VTensorPlot)

    IF(VISIT_PYTHON_SCRIPTING)
        SET(LIBS_SOURCES
            TensorScriptingPluginInfo.C
            PyTensorAttributes.C
            ${COMMON_SOURCES}
        )
        ADD_LIBRARY(STensorPlot ${LIBS_SOURCES})
        IF(WIN32)
            # This prevents python from #defining snprintf as _snprintf
            SET_TARGET_PROPERTIES(STensorPlot PROPERTIES COMPILE_DEFINITIONS HAVE_SNPRINTF)
        ENDIF()
        TARGET_LINK_LIBRARIES(STensorPlot visitcommon visitpy ${PYTHON_LIBRARY})
        SET(INSTALLTARGETS ${INSTALLTARGETS} STensorPlot)
    ENDIF(VISIT_PYTHON_SCRIPTING)

    IF(VISIT_JAVA)
        FILE(COPY TensorAttributes.java DESTINATION ${JavaClient_BINARY_DIR}/src/plots)
        ADD_CUSTOM_TARGET(JavaTensor ALL ${Java_JAVAC_EXECUTABLE} ${VISIT_Java_FLAGS} -d ${JavaClient_BINARY_DIR} -classpath ${JavaClient_BINARY_DIR} -sourcepath ${JavaClient_BINARY_DIR} TensorAttributes.java
            DEPENDS JavaClient
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    ENDIF(VISIT_JAVA)
ENDIF(NOT VISIT_SERVER_COMPONENTS_ONLY AND NOT VISIT_ENGINE_ONLY AND NOT VISIT_DBIO_ONLY)

ADD_LIBRARY(ETensorPlot_ser ${LIBE_SOURCES})
TARGET_LINK_LIBRARIES(ETensorPlot_ser visitcommon avtplotter_ser avtpipeline_ser )
SET(INSTALLTARGETS ${INSTALLTARGETS} ETensorPlot_ser)
ADD_TARGET_DEFINITIONS(ETensorPlot_ser ENGINE)

IF(VISIT_PARALLEL)
    ADD_PARALLEL_LIBRARY(ETensorPlot_par ${LIBE_SOURCES})
    TARGET_LINK_LIBRARIES(ETensorPlot_par visitcommon avtplotter_par avtpipeline_par )
    SET(INSTALLTARGETS ${INSTALLTARGETS} ETensorPlot_par)
    ADD_TARGET_DEFINITIONS(ETensorPlot_par ENGINE)
ENDIF(VISIT_PARALLEL)

VISIT_INSTALL_PLOT_PLUGINS(${INSTALLTARGETS})
VISIT_PLUGIN_TARGET_RTOD(plots ${INSTALLTARGETS})
VISIT_PLUGIN_TARGET_FOLDER(plots Tensor ${INSTALLTARGETS})

