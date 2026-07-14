<script lang="ts">
  let className = "";
  export { className as class };
</script>

<section class="ui-card {className}">
  <slot />
</section>
