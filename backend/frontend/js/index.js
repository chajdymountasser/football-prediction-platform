// ------------------- MOCK DATA -------------------

const latestItems = [
  {
    title: "Win a signed Erling Haaland Man City shirt",
    category: "Competitions",
    thumb: "/static/image/example3.png",
    href: "#"
  },
  {
    title: "Select your Guinness Goal of the Month",
    category: "Awards",
    thumb: "/static/image/example2.png",
    href: "#"
  },
  {
    title: "Choose your Barclays Manager of the Month",
    category: "Awards",
    thumb: "/static/image/example4.jpg",
    href: "#"
  },
  {
    title: "Pick your EA SPORTS Player of the Month",
    category: "Awards",
    thumb: "/static/image/example5.jpg",
    href: "#"
  },
  {
    title: "What we've learned from Matchweek 14 so far",
    category: "Features",
    thumb: "/static/image/example6.jpg",
    href: "#"
  }
];

const scoresData = [
  {
    dateLabel: "Sat 31 Jan",
    matches: [
      {
        home: "Brighton",
        away: "Everton",
        homeBadge: "/static/icons/PL/brighton.svg",
        awayBadge: "/static/icons/PL/everton.svg",
        homeScore: 0,
        awayScore: 1,
        status: "FT"
      },
      {
        home: "Leeds",
        away: "Arsenal",
        homeBadge: "/static/icons/PL/leeds.svg",
        awayBadge: "/static/icons/PL/arsenal.svg",
        homeScore: 1,
        awayScore: 3,
        status: "FT"
      },
      {
        home: "Chelsea",
        away: "West Ham",
        homeBadge: "/static/icons/PL/chelsea.svg",
        awayBadge: "/static/icons/PL/westham.svg",
        homeScore: 2,
        awayScore: 0,
        status: "FT"
      }
    ]
  },
  {
    dateLabel: "Sun 1 Feb",
    matches: [
      {
        home: "Man Utd",
        away: "Fulham",
        homeBadge: "/static/icons/PL/manutd.svg",
        awayBadge: "/static/icons/PL/fulham.svg",
        homeScore: 1,
        awayScore: 1,
        status: "FT"
      },
      {
        home: "Spurs",
        away: "Man City",
        homeBadge: "/static/icons/PL/spurs.svg",
        awayBadge: "/static/icons/PL/mancity.svg",
        homeScore: 1,
        awayScore: 3,
        status: "FT"
      }
    ]
  }
];

const featuresItems = [
  {
    title: "Outstanding De Ligt is now a central figure",
    category: "Analysis",
    image: "/static/image/example1.jpg",
    href: "#"
  },
  {
    title: "Key questions for final days of January transfer window",
    category: "Features",
    image: "/static/image/example11.jpg",
    href: "#"
  },
  {
    title: "Arsenal show strength in depth",
    category: "Features",
    image: "/static/image/hero3.jpg",
    href: "#"
  },
  {
    title: "The FASTEST players in Premier League history",
    category: "Analysis",
    image: "/static/image/example10.jpg",
    href: "#"
  },
  {
    title: "Pick YOUR Premier League Team of the Week",
    category: "Features",
    image: "/static/image/example9.jpg",
    href: "#"
  }
];

// ------------------- RENDERERS -------------------

function renderLatest(items) {
  const wrap = document.getElementById("latestList");
  wrap.innerHTML = items.map(item => `
    <a class="middle-item" href="${item.href}">
      <div>
        <p class="middle-item__headline">${item.title}</p>
        <span class="middle-item__sub">${item.category}</span>
      </div>
      <img class="middle-item__thumb" src="${item.thumb}">
    </a>
  `).join("");
}

function renderScores(blocks) {
  const wrap = document.getElementById("scoresList");
  wrap.innerHTML = blocks.map(block => `
    <div class="scores-date">${block.dateLabel}</div>
    ${block.matches.map(m => `
      <div class="score-row">
        <div class="team left">
          <span class="team-name">${m.home}</span>
          <img class="team-badge" src="${m.homeBadge}">
        </div>
        <div class="score-center">
          <div class="score-pill">${m.homeScore} - ${m.awayScore}</div>
          <div class="status">${m.status}</div>
        </div>
        <div class="team right">
          <img class="team-badge" src="${m.awayBadge}">
          <span class="team-name">${m.away}</span>
        </div>
      </div>
    `).join("")}
  `).join("");
}

function renderFeatures(items) {
  const wrap = document.getElementById("featuresRow");
  wrap.innerHTML = items.map(item => `
    <article class="feature-card">
      <a href="${item.href}">
        <img src="${item.image}">
        <div class="feature-body">
          <h3>${item.title}</h3>
          <span class="feature-sub">${item.category}</span>
        </div>
      </a>
    </article>
  `).join("");
}

// ------------------- INIT -------------------

renderLatest(latestItems);
renderScores(scoresData);
renderFeatures(featuresItems);
