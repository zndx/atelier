// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

// Simple script to make D2 diagrams clickable
// Opens diagram in new tab when clicked
(function() {
    'use strict';

    function makeDiagramsClickable() {
        // Find all D2 diagram images
        var images = document.querySelectorAll('img[src*="d2/"]');

        for (var i = 0; i < images.length; i++) {
            var img = images[i];

            // Make cursor pointer
            img.style.cursor = 'zoom-in';
            img.style.transition = 'opacity 0.2s';

            // Add click handler to open in new tab
            img.onclick = function() {
                window.open(this.src, '_blank');
            };

            // Add hover effect
            img.onmouseenter = function() {
                this.style.opacity = '0.85';
            };

            img.onmouseleave = function() {
                this.style.opacity = '1';
            };
        }
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', makeDiagramsClickable);
    } else {
        makeDiagramsClickable();
    }
})();
