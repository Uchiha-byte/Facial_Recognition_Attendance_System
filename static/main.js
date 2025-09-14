document.addEventListener('DOMContentLoaded', () => {
    // Show loading spinner during async operations
    document.querySelectorAll('a.button, button[type="submit"]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelector('.loading').style.display = 'flex';
      });
    });
  
    // Modal handling
    const modals = {
      init() {
        document.addEventListener('click', e => {
          if (e.target.closest('.danger')) {
            this.showConfirmation(e.target.closest('form'));
          }
          if (e.target.classList.contains('modal')) {
            this.close();
          }
        });
      },
  
      showConfirmation(form) {
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
          <div class="modal-content">
            <h3>Are you sure?</h3>
            <p>This action cannot be undone.</p>
            <div class="modal-buttons">
              <button class="button danger confirm">Confirm</button>
              <button class="button primary cancel">Cancel</button>
            </div>
          </div>
        `;
        
        modal.querySelector('.confirm').addEventListener('click', () => {
          form.submit();
          this.close();
        });
        
        modal.querySelector('.cancel').addEventListener('click', () => this.close());
        
        document.body.appendChild(modal);
        modal.style.display = 'flex';
      },
  
      close() {
        const modal = document.querySelector('.modal');
        if (modal) {
          modal.remove();
        }
      }
    };
  
    modals.init();
  
    // Form validation
    document.querySelectorAll('form').forEach(form => {
      form.addEventListener('submit', e => {
        const inputs = form.querySelectorAll('input[required]');
        let valid = true;
        
        inputs.forEach(input => {
          if (!input.value.trim()) {
            input.style.borderColor = 'var(--danger)';
            valid = false;
          }
        });
        
        if (!valid) {
          e.preventDefault();
          alert('Please fill in all required fields');
        }
      });
    });
  
    // Real-time clock for attendance
    const updateClock = () => {
      const now = new Date();
      document.querySelectorAll('.live-clock').forEach(element => {
        element.textContent = now.toLocaleTimeString();
      });
    };
    setInterval(updateClock, 1000);
  });
  
  // Initialize app
  document.addEventListener('DOMContentLoaded', function() {
    console.log('Facial Recognition Attendance System loaded');
  });
