# DO NOT EDIT THIS FILE! THIS FILE IS AUTOMATICALLY GENERATED BY xml2cmake
PROJECT(CartographicProjection_operator)

INCLUDE(${VISIT_SOURCE_DIR}/CMake/PluginMacros.cmake)

ADD_OPERATOR_CODE_GEN_TARGETS(CartographicProjection)

SET(COMMON_SOURCES
CartographicProjectionPluginInfo.C
CartographicProjectionCommonPluginInfo.C
CartographicProjectionAttributes.C
)

SET(LIBI_SOURCES
CartographicProjectionPluginInfo.C
)

SET(LIBG_SOURCES
CartographicProjectionGUIPluginInfo.C
QvisCartographicProjectionWindow.C
${COMMON_SOURCES}
)
SET(LIBG_MOC_SOURCES
QvisCartographicProjectionWindow.h
)

SET(LIBV_SOURCES
CartographicProjectionViewerEnginePluginInfo.C
CartographicProjectionViewerPluginInfo.C
${COMMON_SOURCES}
)

SET(LIBE_SOURCES
CartographicProjectionViewerEnginePluginInfo.C
CartographicProjectionEnginePluginInfo.C
avtCartographicProjectionFilter.C
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
${VISIT_INCLUDE_DIR}/avt/Expressions/Abstract
${VISIT_INCLUDE_DIR}/avt/Expressions/CMFE
${VISIT_INCLUDE_DIR}/avt/Expressions/Conditional
${VISIT_INCLUDE_DIR}/avt/Expressions/Derivations
${VISIT_INCLUDE_DIR}/avt/Expressions/General
${VISIT_INCLUDE_DIR}/avt/Expressions/ImageProcessing
${VISIT_INCLUDE_DIR}/avt/Expressions/Management
${VISIT_INCLUDE_DIR}/avt/Expressions/Math
${VISIT_INCLUDE_DIR}/avt/Expressions/MeshQuality
${VISIT_INCLUDE_DIR}/avt/Expressions/TimeIterators
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
${VISIT_INCLUDE_DIR}/mdserver/proxy
${VISIT_INCLUDE_DIR}/mdserver/rpc
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

ADD_LIBRARY(ICartographicProjectionOperator ${LIBI_SOURCES})
TARGET_LINK_LIBRARIES(ICartographicProjectionOperator visitcommon)
SET(INSTALLTARGETS ICartographicProjectionOperator)

IF(NOT VISIT_SERVER_COMPONENTS_ONLY AND NOT VISIT_ENGINE_ONLY AND NOT VISIT_DBIO_ONLY)
    QT_WRAP_CPP(GCartographicProjectionOperator LIBG_SOURCES ${LIBG_MOC_SOURCES})
    ADD_LIBRARY(GCartographicProjectionOperator ${LIBG_SOURCES})
    TARGET_LINK_LIBRARIES(GCartographicProjectionOperator visitcommon gui )

    ADD_LIBRARY(VCartographicProjectionOperator ${LIBV_SOURCES})
    TARGET_LINK_LIBRARIES(VCartographicProjectionOperator visitcommon viewer )
    SET(INSTALLTARGETS ${INSTALLTARGETS} GCartographicProjectionOperator VCartographicProjectionOperator)

    IF(VISIT_PYTHON_SCRIPTING)
        SET(LIBS_SOURCES
            CartographicProjectionScriptingPluginInfo.C
            PyCartographicProjectionAttributes.C
            ${COMMON_SOURCES}
        )
        ADD_LIBRARY(SCartographicProjectionOperator ${LIBS_SOURCES})
        IF(WIN32)
            # This prevents python from #defining snprintf as _snprintf
            SET_TARGET_PROPERTIES(SCartographicProjectionOperator PROPERTIES COMPILE_DEFINITIONS HAVE_SNPRINTF)
        ENDIF()
        TARGET_LINK_LIBRARIES(SCartographicProjectionOperator visitcommon visitpy ${PYTHON_LIBRARY})
        SET(INSTALLTARGETS ${INSTALLTARGETS} SCartographicProjectionOperator)
    ENDIF(VISIT_PYTHON_SCRIPTING)

    IF(VISIT_JAVA)
        FILE(COPY CartographicProjectionAttributes.java DESTINATION ${JavaClient_BINARY_DIR}/src/operators)
        ADD_CUSTOM_TARGET(JavaCartographicProjection ALL ${Java_JAVAC_EXECUTABLE} ${VISIT_Java_FLAGS} -d ${JavaClient_BINARY_DIR} -classpath ${JavaClient_BINARY_DIR} -sourcepath ${JavaClient_BINARY_DIR} CartographicProjectionAttributes.java
            DEPENDS JavaClient
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    ENDIF(VISIT_JAVA)
ENDIF(NOT VISIT_SERVER_COMPONENTS_ONLY AND NOT VISIT_ENGINE_ONLY AND NOT VISIT_DBIO_ONLY)

ADD_LIBRARY(ECartographicProjectionOperator_ser ${LIBE_SOURCES})
TARGET_LINK_LIBRARIES(ECartographicProjectionOperator_ser visitcommon avtexpressions_ser avtfilters_ser avtpipeline_ser vtkGeovisCore )
SET(INSTALLTARGETS ${INSTALLTARGETS} ECartographicProjectionOperator_ser)

IF(VISIT_PARALLEL)
    ADD_PARALLEL_LIBRARY(ECartographicProjectionOperator_par ${LIBE_SOURCES})
    TARGET_LINK_LIBRARIES(ECartographicProjectionOperator_par visitcommon avtexpressions_par avtfilters_par avtpipeline_par vtkGeovisCore )
    SET(INSTALLTARGETS ${INSTALLTARGETS} ECartographicProjectionOperator_par)
ENDIF(VISIT_PARALLEL)

VISIT_INSTALL_OPERATOR_PLUGINS(${INSTALLTARGETS})
VISIT_PLUGIN_TARGET_RTOD(operators ${INSTALLTARGETS})
VISIT_PLUGIN_TARGET_FOLDER(operators CartographicProjection ${INSTALLTARGETS})

