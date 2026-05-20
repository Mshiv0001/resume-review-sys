// Simple theme toggle. Stores selection in localStorage and applies to <html data-theme>
(function(){
  const key = 'insdash_theme';
  const root = document.documentElement;
  const btnId = 'themeToggleBtn';

  function applyTheme(theme){
    if(theme === 'light') root.setAttribute('data-theme','light');
    else root.removeAttribute('data-theme');
  }

  function toggle(){
    const cur = localStorage.getItem(key) || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    localStorage.setItem(key,next);
    applyTheme(next);
    updateButton(next);
  }

  function updateButton(theme){
    const btn = document.getElementById(btnId);
    if(!btn) return;
    btn.innerHTML = theme === 'light' ? '🌙' : '☀️';
    btn.title = theme === 'light' ? 'Switch to dark' : 'Switch to light';
  }

  document.addEventListener('DOMContentLoaded',()=>{
    const saved = localStorage.getItem(key) || 'dark';
    applyTheme(saved);
    updateButton(saved);
    const btn = document.getElementById(btnId);
    if(btn) btn.addEventListener('click',toggle);
  });
})();
