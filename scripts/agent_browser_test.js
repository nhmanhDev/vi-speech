// Test voice agent trong Chromium headless với micro giả (file wav). Chạy server trước:
//   python -m vispeech.agent.server --asr ... --tts ...
//   node scripts/agent_browser_test.js /path/to/mic.wav
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    '--use-file-for-fake-audio-capture=' + (process.argv[2] || 'mic.wav') + '', '--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('console:', m.text()));
  await page.goto('http://localhost:8765/');
  await page.click('#btn');
  await page.waitForTimeout(15000);
  
  const msgs = await page.$$eval('#log > *', els => els.map(e => e.className + ' | ' + e.textContent));
  console.log(msgs.reverse().join('\n'));
  console.log('state:', await page.textContent('#stateText'));
  await page.screenshot({ path: 'agent_page.png' });
  await browser.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });
