// =================== ROOT.JS (universal navbar/drawer) ===================
document.addEventListener("DOMContentLoaded", () => {
  const drawer = document.getElementById("navDrawer");
  const backdrop = document.getElementById("navBackdrop");
  const openBtn = document.querySelector(".top-nav .icon-btn");
  const closeBtn = drawer?.querySelector(".nav-drawer__close");

  if (!drawer || !backdrop || !openBtn) return;

  const openDrawer = () => {
    drawer.classList.add("is-open");
    backdrop.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    backdrop.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  };

  const closeDrawer = () => {
    drawer.classList.remove("is-open");
    backdrop.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    backdrop.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  };

  openBtn.addEventListener("click", openDrawer);
  closeBtn?.addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);

  // close on ESC
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer.classList.contains("is-open")) {
      closeDrawer();
    }
  });

  // optional: highlight active page link automatically
  const links = drawer.querySelectorAll(".nav-drawer__links a");
  const current = location.pathname;
  links.forEach(a => {
    const href = a.getAttribute("href");
    if (href === current) a.classList.add("active");
  });
  
});
