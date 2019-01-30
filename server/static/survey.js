window.onload = function() {
    updateShowPhenomena();

    var options = document.getElementsByClassName('mainBox');
    for (option of options) {
        updateOptionIcon(option);
    }

    var subLists = document.querySelectorAll('ul.subList');
    subLists.forEach(subList => {
        var shouldBeVisible = false;
        var subOptionChildren = subList.querySelectorAll('input.subOption');
        for (subOption of subOptionChildren) {
            updateSuboptionIcon(subOption);
        }
        for (subOption of subOptionChildren) {
            if (subOption.checked == true) {
                shouldBeVisible = true;
                break; //return from inner loop
            }
        }
        if (!shouldBeVisible) {
            subList.style.display = 'none';
        }
    });

    var ttip = document.getElementById("ttip");
    var canvas = document.getElementById("ttip-canvas");
    var canvasContext = canvas.getContext("2d");

    var ttipImage = new Image();
    var _rectX = 0;
    var _rectY = 0;
    var _rectWidth = 0;
    var _rectHeight = 0;

    ttipImage.onload = function() {
        var factor = 400 / ttipImage.height;
        canvas.width = ttipImage.width * factor;
        canvas.height = 400;
        console.log(canvas.style.height);
        canvasContext.drawImage(ttipImage, 0, 0, ttipImage.width * factor, 400);
        // draw rectangle
        canvasContext.rect(_rectX * factor, _rectY * factor , _rectWidth * factor, _rectHeight * factor);
        canvasContext.lineWidth = 3;
        canvasContext.strokeStyle = "yellow";
        canvasContext.stroke();
    };

    window.onmousemove = function (e) {
        var x = (e.clientX + 20) + 'px';
        var y = (e.clientY + 20) + 'px';
        ttip.style.left = x;
        ttip.style.top = y;
        console.log('moved');
    };

    showttip = function(imglink, rectX, rectY, rectWidth, rectHeight) {
        console.log("x: " + rectX + ", y: " + rectY + ", width: " +rectWidth + ", height: " +rectHeight);
        _rectY = rectX;
        _rectX = rectY;
        _rectWidth = rectWidth;
        _rectHeight = rectHeight;
        ttip.style.display = "block";
        ttipImage.setAttribute("src", imglink);
    };

    hidettip = function() {
        canvasContext.clearRect(0, 0, canvas.width, canvas.height);
        ttip.style.display = "none";
    }
};

showttip = function() {}
hidettip = function() {}

function updateShowPhenomena() {
    if (document.getElementById('shows_phenomena_true').checked) {
        document.getElementById('phenomena_desc').style.display = 'block';
    } else {
        document.getElementById('phenomena_desc').style.display = 'none';
    }
}

function optionClicked(elem) {
    updateOptionIcon(elem);
    var siblings = getSiblings(elem);
    for (s of siblings) {
        if (s.tagName != 'LABEL') {
            if (elem.checked) {
                s.style.display = 'block';
            } else {
                s.style.display = 'none';
                s.querySelectorAll('input.subOption').forEach(subOpt => {
                    subOpt.checked = false;
                    updateSuboptionIcon(subOpt);
                })
            }
        }
    }
}

function updateSuboptionText(elem) {
    var label = elem.parentElement;
    if (elem.checked === true) {
        label.classList.add("label-checked");
    } else {
        if (label.classList.contains("label-checked")) {
            label.classList.remove("label-checked");
        }
    }
}

function updateOptionIcon(elem) {
    var siblings = getSiblings(elem);
    for (s of siblings) {
        if (s.tagName == 'LABEL') {
            var octicons = s.getElementsByClassName('octicon');
            if (elem.checked) {
                setOcticonsChecked(octicons);
            } else {
                setOcticonsUnchecked(octicons);
            }
        }
    }
}

function updateSuboptionIcon(elem) {
    if (elem.checked === true) {
        var octicons = elem.parentElement.getElementsByClassName('octicon');
        setOcticonsChecked(octicons);
    } else {
        var octicons = elem.parentElement.getElementsByClassName('octicon');
        setOcticonsUnchecked(octicons);
    }
    updateSuboptionText(elem);
}

function setOcticonsChecked(octicons) {
    for (octicon of octicons) {
        if (octicon.classList.contains("octicon-chevron-right")) {
            octicon.classList.remove("octicon-chevron-right");
            octicon.classList.add("octicon-chevron-down");
        } else if (octicon.classList.contains("octicon-x")) {
            octicon.classList.remove("octicon-x");
            octicon.classList.add("octicon-check");
        }
    }
}

function setOcticonsUnchecked(octicons) {
    for (octicon of octicons) {
        if (octicon.classList.contains("octicon-chevron-down")) {
            octicon.classList.remove("octicon-chevron-down");
            octicon.classList.add("octicon-chevron-right");
        } else if (octicon.classList.contains("octicon-check")) {
            octicon.classList.remove("octicon-check");
            octicon.classList.add("octicon-x");
        }
    }
}

function getSiblings (elem) {
    var siblings = [];
    var sibling = elem.parentNode.firstChild;
    for (; sibling; sibling = sibling.nextSibling) {
        if (sibling.nodeType !== 1 || sibling === elem) continue;
        siblings.push(sibling);
    }
    return siblings;
}