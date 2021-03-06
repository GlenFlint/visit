// Copyright (c) Lawrence Livermore National Security, LLC and other VisIt
// Project developers.  See the top-level LICENSE file for dates and other
// details.  No copyright assignment is required to contribute to VisIt.

#include <QvisColorTableButton.h>
#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QDesktopWidget>
#include <QMenu>
#include <QPainter>
#include <QPixmap>

#include <ColorTableAttributes.h>
#include <ColorControlPointList.h>

#define ICON_NX 32
#define ICON_NY 16

//
// Static members.
//

int           QvisColorTableButton::numInstances = 0;
QMenu        *QvisColorTableButton::colorTableMenu = 0;
QActionGroup *QvisColorTableButton::colorTableMenuActionGroup = 0;
QvisColorTableButton::ColorTableButtonVector QvisColorTableButton::buttons;
QStringList  QvisColorTableButton::colorTableNames;
QMap<QString,QStringList>  QvisColorTableButton::mappedColorTableNames;
bool        QvisColorTableButton::popupHasEntries = false;
ColorTableAttributes *QvisColorTableButton::colorTableAtts = NULL;

// ****************************************************************************
// Method: QvisColorTableButton::QvisColorTableButton
//
// Purpose: 
//   Constructor for the QvisColorTableButton class.
//
// Arguments:
//   parent : The parent widget.
//   name   : The name to associate with this widget.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:06:13 PST 2001
//
// Modifications:
//   Brad Whitlock, Thu Feb 14 13:38:42 PST 2002
//   Added code to count the number of instances.
//
//   Brad Whitlock, Fri May  9 11:23:57 PDT 2008
//   Qt 4.
//
// ****************************************************************************

QvisColorTableButton::QvisColorTableButton(QWidget *parent) :
    QPushButton(parent), colorTable("Default")
{
    // Increase the instance count.
    ++numInstances;

    // Create the button's color table popup menu.
    if(colorTableMenu == 0)
    {
        colorTableMenuActionGroup = new QActionGroup(0);

        colorTableMenu = new QMenu(0);
        colorTableMenuActionGroup->addAction(colorTableMenu->addAction("Default"));
        colorTableMenu->addSeparator();
    }
    buttons.push_back(this);

    // Make the popup active when this button is clicked.
    connect(this, SIGNAL(pressed()), this, SLOT(popupPressed()));

    setText(colorTable);
    setIconSize(QSize(ICON_NX,ICON_NY));
}

// ****************************************************************************
// Method: QvisColorTableButton::~QvisColorTableButton
//
// Purpose: 
//   This is the destructor for the QvisColorTableButton class.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:06:57 PST 2001
//
// Modifications:
//   Brad Whitlock, Thu Feb 14 13:31:46 PST 2002
//   Deleted the popup menu if it exists.
//
// ****************************************************************************

QvisColorTableButton::~QvisColorTableButton()
{
    // Decrease the instance count.
    --numInstances;

    // Remove the "this" pointer from the vector.
    size_t index = 0;
    bool notFound = true;
    for(size_t i = 0; i < buttons.size() && notFound; ++i)
    {
        if(this == buttons[i])
        {
            notFound = false;
            index = i;
        }
    }

    // If the pointer was found, shift the pointers in the vector and pop the
    // last element.
    if(!notFound)
    {
        for(size_t i = index; i < buttons.size() - 1; ++i)
            buttons[i] = buttons[i + 1];
        buttons.pop_back();
    }

    if(numInstances == 0)
    {
        if(colorTableMenuActionGroup != 0)
        {
            delete colorTableMenuActionGroup;
            colorTableMenuActionGroup = 0;
        }

        // Delete the popup menu if it exists because it will not be deleted
        // unless we do it since it is a parentless widget.
        if(colorTableMenu != 0)
        {
            delete colorTableMenu;
            colorTableMenu = 0;
        }

        // Delete the color table names.
        colorTableNames.clear();
        mappedColorTableNames.clear();
    }
}

// ****************************************************************************
// Method: QvisColorTableButton::sizeHint
//
// Purpose: 
//   Returns the widget's preferred size.
//
// Returns:    The widget's preferred size.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:07:23 PST 2001
//
// Modifications:
//   
// ****************************************************************************

QSize
QvisColorTableButton::sizeHint() const
{
     return QSize(125, 40).expandedTo(QApplication::globalStrut());
}

// ****************************************************************************
// Method: QvisColorTableButton::sizePolicy
//
// Purpose: 
//   Returns the widget's size policy -- how allows itself to be resized.
//
// Returns:    The widget's size policy.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:07:55 PST 2001
//
// Modifications:
//   
// ****************************************************************************

QSizePolicy
QvisColorTableButton::sizePolicy() const
{
    return QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

// ****************************************************************************
// Method: QvisColorTableButton::useDefaultColorTable
//
// Purpose: 
//   Tells the widget to use the default color table.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:08:42 PST 2001
//
// Modifications:
//   Brad Whitlock, Tue Jan 17 11:41:44 PDT 2006
//   Added a tooltip so long color table names can be put in a tooltip.
//   
// ****************************************************************************

void
QvisColorTableButton::useDefaultColorTable()
{
    colorTable = QString("Default");
    setText(colorTable);
    setToolTip(colorTable);
    setIcon(QIcon());
}

// ****************************************************************************
// Method: QvisColorTableButton::setColorTable
//
// Purpose: 
//   Tells the widget to use a specified color table.
//
// Arguments:
//   ctName : The name of the color table to use.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:09:09 PST 2001
//
// Modifications:
//   Brad Whitlock, Fri Feb 15 10:15:43 PDT 2002
//   Made it set the menu text to "Default" if no color table is found.
//
//   Brad Whitlock, Tue Jan 17 11:41:44 PDT 2006
//   Added a tooltip so long color table names can be put in a tooltip.
//
// ****************************************************************************

void
QvisColorTableButton::setColorTable(const QString &ctName)
{
    if(getColorTableIndex(ctName) != -1)
    {
        colorTable = ctName;
        setText(colorTable);
        setToolTip(colorTable);
        setIcon(getIcon(ctName));
    }
    else
    {
        QString def("Default");
        setText(def);
        setToolTip(def);
        setIcon(QIcon());
    }
}

// ****************************************************************************
// Method: QvisColorTableButton::getColorTable
//
// Purpose: 
//   Returns the name of the widget's color table.
//
// Returns:    The name of the widget's color table.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:09:46 PST 2001
//
// Modifications:
//   
// ****************************************************************************

const QString &
QvisColorTableButton::getColorTable() const
{
    return colorTable;
}

//
// Qt slot functions.
//

// ****************************************************************************
// Method: QvisColorTableButton::popupPressed
//
// Purpose: 
//   This is a Qt slot function that pops up the color table popup menu when
//   the button is pressed.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:10:16 PST 2001
//
// Modifications:
//   
// ****************************************************************************

void
QvisColorTableButton::popupPressed()
{
    if(isDown() && colorTableMenu)
    {
        // If the popup menu does not have anything in it, fill it up.
        if(!popupHasEntries)
            regeneratePopupMenu();

        QPoint p1(mapToGlobal(rect().bottomLeft()));
        QPoint p2(mapToGlobal(rect().topRight()));
        QPoint buttonMiddle(p1.x() + ((p2.x() - p1.x()) >> 1),
                            p1.y() + ((p2.y() - p1.y()) >> 1));

        // Disconnect all other color table buttons.
        for(size_t i = 0; i < buttons.size(); ++i)
        {
            disconnect(colorTableMenuActionGroup, SIGNAL(triggered(QAction *)),
                       buttons[i], SLOT(colorTableSelected(QAction *)));
        }

        // Connect this colorbutton to the popup menu.
        connect(colorTableMenuActionGroup, SIGNAL(triggered(QAction *)),
                this, SLOT(colorTableSelected(QAction *)));

        // Figure out a good place to popup the menu.
        int menuW = colorTableMenu->sizeHint().width();
        int menuH = colorTableMenu->sizeHint().height();
        int menuX = buttonMiddle.x();
        int menuY = buttonMiddle.y() - (menuH >> 1);

        // Fix the X dimension.
        if(menuX < 0)
           menuX = 0;
        else if(menuX + menuW > QApplication::desktop()->width())
           menuX -= (menuW + 5);

        // Fix the Y dimension.
        if(menuY < 0)
           menuY = 0;
        else if(menuY + menuH > QApplication::desktop()->height())
           menuY -= ((menuY + menuH) - QApplication::desktop()->height());

        // Show the popup menu.         
        colorTableMenu->exec(QPoint(menuX, menuY));
        setDown(false);
    }
}

// ****************************************************************************
// Method: QvisColorTableButton::colorTableSelected
//
// Purpose: 
//   This is a Qt slot function that is called when a color table has been
//   selected from the popup menu. The widget then emits a selectedColorTable
//   signal.
//
// Arguments:
//   index : The index of the color table chosen from the popup menu.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:11:06 PST 2001
//
// Modifications:
//   Brad Whitlock, Tue Jan 17 11:41:44 PDT 2006
//   Added a tooltip so long color table names can be put in a tooltip.
//
//   Brad Whitlock, Fri May  9 11:39:40 PDT 2008
//   Qt 4.
//
//   Kathleen Biagas, Mon Aug  4 15:54:14 PDT 2014
//   Handle grouping.
//
// ****************************************************************************

void
QvisColorTableButton::colorTableSelected(QAction *action)
{
    int index = colorTableMenuActionGroup->actions().indexOf(action);

    if(index == 0)
    {
        QString def("Default");
        emit selectedColorTable(true, def);
        setText(def);
        setToolTip(def);
        setIcon(QIcon());
    }
    else
    {
        QString ctName;
        if (!colorTableAtts->GetGroupingFlag() || mappedColorTableNames.count() == 1)
        {
            ctName = colorTableNames.at(index-1);
        }
        else
        {
            int count=1, N=0;
            QMap<QString, QStringList>::const_iterator iter;
            for(iter = mappedColorTableNames.constBegin();
                iter != mappedColorTableNames.constEnd();
                ++iter)
            {
                N = iter.value().size();
                if(index < (count+N))
                {
                    ctName = iter.value().at(index-count);
                    break;
                }
                count += N;
            }
        }

        emit selectedColorTable(false, ctName);
        setText(ctName);
        setIcon(getIcon(ctName));
        setToolTip(ctName);
    }
}

//
// Static methods
//

// ****************************************************************************
// Method: QvisColorTableButton::clearAllColorTables
//
// Purpose: 
//   This is a static method to clear all of the known color tables.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:12:33 PST 2001
//
// Modifications:
//
// ****************************************************************************

void
QvisColorTableButton::clearAllColorTables()
{
    colorTableNames.clear();
    mappedColorTableNames.clear();

    // Clear out the popup menu.
    popupHasEntries = false;
}

// ****************************************************************************
// Method: QvisColorTableButton::addColorTable
//
// Purpose: 
//   This is a static method that tells the widget about a new color table.
//
// Arguments:
//   ctName : The name of the new color table.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:13:09 PST 2001
//
// Modifications:
//   Kathleen Biagas, Mon Aug  4 15:55:26 PDT 2014
//   colorTableNames now a QStringList, so append and sort.
//   Added mappedColorTableNames.
//
// ****************************************************************************

void
QvisColorTableButton::addColorTable(const QString &ctName,
    const QString &ctCategory)
{
    colorTableNames.append(ctName);
    colorTableNames.sort();
    mappedColorTableNames[ctCategory].append(ctName);
    mappedColorTableNames[ctCategory].sort();
}

// ****************************************************************************
// Method: QvisColorTableButton::updateColorTableButtons
//
// Purpose: 
//   This is a static method that iterates through all instances of
//   QvisColorTableButton to make sure that the color table that they use is
//   a valid color table. This will also be used to update their color table
//   pixmaps -- someday.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:13:46 PST 2001
//
// Modifications:
//   
// ****************************************************************************

void
QvisColorTableButton::updateColorTableButtons()
{
    for(size_t i = 0; i < buttons.size(); ++i)
    {
        // If the color table that was being used by the button is no
        // longer in the list, make it use the default.
        if(getColorTableIndex(buttons[i]->getColorTable()) == -1)
        {
            buttons[i]->setText("Default");
            buttons[i]->setColorTable("Default");
        }
        else
            buttons[i]->setIcon(getIcon(buttons[i]->text()));
    }
}

// ****************************************************************************
// Method: QvisColorTableButton::getColorTableIndex
//
// Purpose: 
//   Returns the index of the specified color table in the internal color
//   table list. If the color table is not found, -1 is returned.
//
// Arguments:
//   ctName : The name of the color table to look for.
//
// Returns:    The index of the color table, or -1.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:15:16 PST 2001
//
// Modifications:
//   Kathleen Biagas, Mon Aug  4 15:59:18 PDT 2014
//   Use the indexOf method for QStringList.
//
// ****************************************************************************

int
QvisColorTableButton::getColorTableIndex(const QString &ctName)
{
    return colorTableNames.indexOf(ctName);
}

// ****************************************************************************
// Method: QvisColorTableButton::regeneratePopupMenu
//
// Purpose: 
//   This method is called when the popup menu needs to be regenerated. This
//   happens when the color table list changes.
//
// Programmer: Brad Whitlock
// Creation:   Sat Jun 16 20:16:34 PST 2001
//
// Modifications:
//   Brad Whitlock, Fri May  9 11:21:28 PDT 2008
//   Qt 4.
//
//   Brad Whitlock, Wed Apr 25 13:32:01 PDT 2012
//   Add pixmaps of the color table.
//
//   Kathleen Biagas, Mon Aug  4 15:59:56 PDT 2014
//   Hangle grouping.
//
// ****************************************************************************

void
QvisColorTableButton::regeneratePopupMenu()
{
    // Remove all items and add the default.
    QList<QAction*> actions = colorTableMenuActionGroup->actions();
    for(int i = 0; i < actions.count(); ++i)
        colorTableMenuActionGroup->removeAction(actions[i]);
    colorTableMenu->clear();

    colorTableMenuActionGroup->addAction(colorTableMenu->addAction("Default"));
    colorTableMenu->addSeparator();
    if (!colorTableAtts->GetGroupingFlag() || mappedColorTableNames.count() == 1)
    {
        // Add an item for each color table.
        for(int i = 0; i < colorTableNames.size(); ++i)
        {
            QAction *action = colorTableMenu->addAction(makeIcon(colorTableNames.at(i)), colorTableNames.at(i));
            colorTableMenuActionGroup->addAction(action);
        }
    }
    else
    {
        QMap<QString, QStringList>::const_iterator iter = mappedColorTableNames.constBegin();
        while (iter != mappedColorTableNames.constEnd())
        {
            QMenu *subMenu = colorTableMenu->addMenu(iter.key());
            QStringList ctNames = iter.value();

            // Add an item for each color table.
            for(int i = 0; i < ctNames.size(); ++i)
            {
                QAction *action = subMenu->addAction(makeIcon(ctNames.at(i)), ctNames.at(i));
                colorTableMenuActionGroup->addAction(action);
            }
            ++iter;
        }
    }

    // Indicate that we've added choices to the menu.
    popupHasEntries = true;
}

// ****************************************************************************
// Method: QvisColorTableButton::getIcon
//
// Purpose: 
//   This method gets the existing icon or makes one if necessary.
//
// Programmer: Brad Whitlock
// Creation:   Wed Apr 25 16:04:54 PDT 2012
//
// Modifications:
//
// ****************************************************************************

QIcon
QvisColorTableButton::getIcon(const QString &ctName)
{
    QList<QAction*> a = colorTableMenuActionGroup->actions();
    for(int i = 0; i < a.size(); ++i)
        if(a[i]->text() == ctName)
            return a[i]->icon();

    return makeIcon(ctName);
}

// ****************************************************************************
// Method: QvisColorTableButton::makeIcon
//
// Purpose: 
//   This method makes an icon from the color table definition.
//
// Programmer: Brad Whitlock
// Creation:   Wed Apr 25 16:04:54 PDT 2012
//
// Modifications:
//
// ****************************************************************************

QIcon
QvisColorTableButton::makeIcon(const QString &ctName)
{
    QIcon icon;
    const ColorControlPointList *cTable = NULL;
    if(colorTableAtts != NULL)
        cTable = colorTableAtts->GetColorControlPoints(ctName.toStdString());
    if(cTable != NULL)
    {
        QPixmap pix(ICON_NX, ICON_NY);
        unsigned char rgb[ICON_NX*3];
        cTable->GetColors(rgb, ICON_NX);
        QPainter paint(&pix);
        for(int ii = 0; ii < ICON_NX; ++ii)
        {
            paint.setPen(QPen(QColor((int)rgb[3*ii+0], (int)rgb[3*ii+1], (int)rgb[3*ii+2])));
            paint.drawLine(ii, 0, ii, ICON_NY-1);
        }

        icon = QIcon(pix);
    }

    return icon;
}

// ****************************************************************************
// Method: QvisColorTableButton::setColorTableAttributes
//
// Purpose: 
//   This method sets the color table attributes.
//
// Programmer: Brad Whitlock
// Creation:   Wed Apr 25 16:04:54 PDT 2012
//
// Modifications:
//
// ****************************************************************************

void
QvisColorTableButton::setColorTableAttributes(ColorTableAttributes *cAtts)
{
    colorTableAtts = cAtts;
}
