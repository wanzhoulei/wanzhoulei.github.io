-- Pandoc Lua filter for the notes pages.
--   ::: qa            -> collapsible question; the first paragraph is the question, the rest the answer
--   ::: {.qa asked="Company, source, year"}  -> same, with a badge saying it was reported in a real interview
--   ::: key / ::: pitfall -> callout boxes
-- Also counts questions so the build script can show "N questions".

local qn = 0

local function esc(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

local function qa_block(el)
  local blocks = el.content
  if #blocks == 0 then return nil end
  qn = qn + 1
  local first = blocks[1]
  local q
  if first.t == "Para" or first.t == "Plain" then
    q = first.content
    if #q == 1 and q[1].t == "Strong" then q = q[1].content end
  else
    q = { pandoc.Str("Question") }
  end
  local asked = el.attributes["asked"]
  local id = "q" .. qn
  local out = {}
  table.insert(out, pandoc.RawBlock("html",
    '<details class="qa' .. (asked and " qa-asked" or "") .. '" id="' .. id .. '"><summary>'))
  local summ = { pandoc.RawInline("html", '<span class="qa-n">Q' .. qn .. '</span><span class="qa-q">') }
  for _, i in ipairs(q) do table.insert(summ, i) end
  table.insert(summ, pandoc.RawInline("html", "</span>"))
  if asked then
    table.insert(summ, pandoc.RawInline("html",
      '<span class="qa-badge" title="Reported in a real interview">Asked · ' .. esc(asked) .. "</span>"))
  end
  table.insert(out, pandoc.Plain(summ))
  table.insert(out, pandoc.RawBlock("html", '</summary><div class="qa-body">'))
  for i = 2, #blocks do table.insert(out, blocks[i]) end
  table.insert(out, pandoc.RawBlock("html", "</div></details>"))
  return out
end

local function callout(el, kind, label)
  local out = { pandoc.RawBlock("html",
    '<aside class="callout callout-' .. kind .. '"><p class="callout-label">' .. label .. "</p>") }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock("html", "</aside>"))
  return out
end

function Div(el)
  if el.classes:includes("qa") then return qa_block(el) end
  if el.classes:includes("key") then return callout(el, "key", "Key idea") end
  if el.classes:includes("pitfall") then return callout(el, "pitfall", "Common pitfall") end
  return nil
end

-- Open external links in a new tab.
function Link(el)
  if el.target:match("^https?://") then
    el.attributes["target"] = "_blank"
    el.attributes["rel"] = "noopener"
  end
  return el
end
