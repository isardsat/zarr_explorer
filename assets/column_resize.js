// Column resizing for the #cmp-unified-table DataTable.
// Captures the table's current rendered column widths as defaults, then
// adds drag handles to each header cell so the user can resize columns.
//
// Dash DataTable structure: the column headers (<th>) are placed inside a
// separate <table> from the body cells, and both live inside <tbody>
// (not <thead>). The body cells (<td>) live in their own <table>. We must
// keep widths in sync across both tables.
(function () {
    "use strict";

    const TABLE_ID = "cmp-unified-table";
    const HANDLE_W = 8;
    const MIN_COL_W = 40;

    let columnWidths = []; // idx -> px
    let setupDone = false;

    function setCellWidth(cell, w) {
        const px = w + "px";
        cell.style.width = px;
        cell.style.minWidth = px;
        cell.style.maxWidth = px;
    }

    function getHeaderCells(container) {
        // The <th> cells are the column headers. Dash places them in the
        // first cell-table inside the container.
        return container.querySelectorAll(
            "table.cell-table tbody tr th[data-dash-column]"
        );
    }

    function getBodyCellsByIndex(container, idx) {
        // Return all <td> cells in column `idx` across all body tables.
        // Body tables have rows whose children are tds at the same index.
        const all = [];
        container.querySelectorAll("table.cell-table tbody tr").forEach((tr) => {
            const cells = tr.children;
            if (cells[idx] && cells[idx].tagName === "TD") {
                all.push(cells[idx]);
            }
        });
        return all;
    }

    function applyOneColumn(container, idx, w) {
        const headerCells = getHeaderCells(container);
        if (headerCells[idx]) setCellWidth(headerCells[idx], w);
        getBodyCellsByIndex(container, idx).forEach((c) => setCellWidth(c, w));
        // Update overall table width hints
        const total = columnWidths.reduce((s, x) => s + (x || 0), 0);
        if (total > 0) {
            container.querySelectorAll("table.cell-table").forEach((t) => {
                t.style.width = total + "px";
                t.style.minWidth = total + "px";
            });
        }
    }

    function applyAllWidths(container) {
        const total = columnWidths.reduce((s, x) => s + (x || 0), 0);
        container.querySelectorAll("table.cell-table").forEach((t) => {
            t.style.tableLayout = "fixed";
            if (total > 0) {
                t.style.width = total + "px";
                t.style.minWidth = total + "px";
            }
        });
        const headerCells = getHeaderCells(container);
        headerCells.forEach((th, i) => {
            if (columnWidths[i]) setCellWidth(th, columnWidths[i]);
        });
        for (let i = 0; i < columnWidths.length; i++) {
            if (!columnWidths[i]) continue;
            getBodyCellsByIndex(container, i).forEach((c) =>
                setCellWidth(c, columnWidths[i])
            );
        }
    }

    function attachHandle(th, idx, container) {
        if (th.dataset.colResizerAttached === "true") return;
        th.dataset.colResizerAttached = "true";

        if (getComputedStyle(th).position === "static") {
            th.style.position = "relative";
        }

        const handle = document.createElement("div");
        handle.className = "col-resize-handle";
        handle.style.cssText =
            "position:absolute;right:0;top:0;bottom:0;width:" +
            HANDLE_W +
            "px;cursor:col-resize;user-select:none;z-index:100;background:transparent;";

        handle.addEventListener("mouseenter", () => {
            if (!handle.classList.contains("dragging")) {
                handle.style.background = "rgba(13,110,253,0.25)";
            }
        });
        handle.addEventListener("mouseleave", () => {
            if (!handle.classList.contains("dragging")) {
                handle.style.background = "transparent";
            }
        });

        let dragging = false;
        let startX = 0;
        let startW = 0;

        handle.addEventListener("mousedown", function (e) {
            e.preventDefault();
            e.stopPropagation();
            dragging = true;
            handle.classList.add("dragging");
            startX = e.clientX;
            startW = columnWidths[idx] || th.getBoundingClientRect().width;
            document.body.style.cursor = "col-resize";
            document.body.style.userSelect = "none";
            handle.style.background = "rgba(13,110,253,0.55)";
        });

        document.addEventListener("mousemove", function (e) {
            if (!dragging) return;
            const newW = Math.max(MIN_COL_W, startW + (e.clientX - startX));
            columnWidths[idx] = newW;
            applyOneColumn(container, idx, newW);
        });

        document.addEventListener("mouseup", function () {
            if (!dragging) return;
            dragging = false;
            handle.classList.remove("dragging");
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
            handle.style.background = "transparent";
        });

        th.appendChild(handle);
    }

    function trySetup(container) {
        const headerCells = getHeaderCells(container);
        if (headerCells.length === 0) return false;

        if (!setupDone) {
            const widths = [];
            for (let i = 0; i < headerCells.length; i++) {
                const w = headerCells[i].getBoundingClientRect().width;
                if (w < 10) return false; // not laid out yet
                widths.push(w);
            }
            columnWidths = widths;
            setupDone = true;
        }

        applyAllWidths(container);
        headerCells.forEach((th, idx) => attachHandle(th, idx, container));
        return true;
    }

    let debounceTimer = null;
    function scheduleSetup() {
        if (debounceTimer) return;
        debounceTimer = setTimeout(() => {
            debounceTimer = null;
            const container = document.getElementById(TABLE_ID);
            if (container) trySetup(container);
        }, 120);
    }

    function start() {
        const observer = new MutationObserver(scheduleSetup);
        observer.observe(document.body, { childList: true, subtree: true });
        scheduleSetup();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
